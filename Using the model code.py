# ========================================================
# Seismic waveform prediction model - inference code (parallel acceleration + comprehensive fixes)
# Fix log:
#   1. load_seismic_file: parse line by line; failed lines are filled with NaN placeholders and linear interpolation to preserve array length and time alignment
#   2. save_single_sample_job: time-column format changed to %.4f to avoid jitter caused by insufficient precision of scientific notation
#   3. prepare_inference_data: added consistency checks for component lengths
#   4. inference_report.txt now includes MAE metrics
#   5. each sample is saved to CSV using its own true station file names (per_sample_station_names)
# ========================================================
import os
import glob
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
from joblib import Parallel, delayed
import matplotlib
import matplotlib.font_manager as fm
# Added at the top of the file
import json
import platform
import concurrent.futures
from pathlib import Path
# joblib is already available; simply confirm it exists
matplotlib.use('Agg')

# ========================================================
# Automatic font configuration
# ========================================================
FONT_FILE = 'SimHei.ttf'


def configure_plotting_font():
    if os.path.exists(FONT_FILE):
        fm.fontManager.addfont(FONT_FILE)
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        return True
    else:
        linux_fonts = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'Noto Sans CJK JP']
        for font in linux_fonts:
            if font in [f.name for f in fm.fontManager.ttflist]:
                plt.rcParams['font.sans-serif'] = [font]
                plt.rcParams['axes.unicode_minus'] = False
                return True
        return False


configure_plotting_font()


# ========================================================
# Part 1: Model definition
# ========================================================
def generate_single_visualization(idx, target_sample, pred_sample, sample_name,
                                  station_names, viz_dir):
    if os.path.exists('SimHei.ttf'):
        fm.fontManager.addfont('SimHei.ttf')
        plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    visualize_waveforms(
        target_sample.unsqueeze(0),
        pred_sample.unsqueeze(0),
        sample_idx=0,
        sample_name=sample_name,
        station_names=station_names,
        save_dir=viz_dir
    )

    visualize_frequency_domain(
        target_sample.unsqueeze(0),
        pred_sample.unsqueeze(0),
        sampling_rate=100, min_freq=0.1, max_freq=10,
        sample_idx=0,
        title=f"Spectrum comparison",
        station_names=station_names,
        save_path=f"{viz_dir}/spectrum_{sample_name.split('_')[-1]}.png"
    )
    return 1


class TemporalAttentionPool(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        scores = self.attention(x)
        attention_weights = torch.softmax(scores, dim=1)
        weighted_sum = torch.sum(x * attention_weights, dim=1)
        return weighted_sum


class EnhancedAmplitudeNetwork(nn.Module):
    def __init__(self, lstm_feature_size, amp_feature_size, output_size, hidden_size=128):
        super(EnhancedAmplitudeNetwork, self).__init__()
        self.attention_extractor = TemporalAttentionPool(lstm_feature_size)
        lstm_expanded_size = lstm_feature_size

        self.fusion_network = nn.Sequential(
            nn.Linear(lstm_expanded_size + amp_feature_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size)
        )

        self.scaling = nn.Parameter(torch.ones(1, output_size) * 1.5)
        self.bias = nn.Parameter(torch.zeros(1, output_size))

    def forward(self, lstm_features, amp_features):
        combined_lstm = self.attention_extractor(lstm_features)
        combined_features = torch.cat([combined_lstm, amp_features], dim=1)
        raw_output = self.fusion_network(combined_features)
        calibrated_output = raw_output * self.scaling + self.bias
        return calibrated_output


class ConvLSTM3DCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, station_dim, component_dim):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.station_dim = station_dim
        self.component_dim = component_dim

        self.Wxi = nn.Conv3d(input_channels, hidden_channels, kernel_size, padding='same', bias=True)
        self.Whi = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding='same', bias=True)
        self.Wxf = nn.Conv3d(input_channels, hidden_channels, kernel_size, padding='same', bias=True)
        self.Whf = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding='same', bias=True)
        self.Wxc = nn.Conv3d(input_channels, hidden_channels, kernel_size, padding='same', bias=True)
        self.Whc = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding='same', bias=True)
        self.Wxo = nn.Conv3d(input_channels, hidden_channels, kernel_size, padding='same', bias=True)
        self.Who = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding='same', bias=True)

        self.Wci = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1, 1))
        self.Wcf = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1, 1))
        self.Wco = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1, 1))

    def forward(self, x, h, c):
        i = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        f = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        g = torch.tanh(self.Wxc(x) + self.Whc(h))
        cc = f * c + i * g
        o = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        ch = o * torch.tanh(cc)
        return ch, cc


class SlidingWindowConvLSTM3D(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, station_dim, component_dim,
                 num_layers=1, window_size=50, stride=10):
        super(SlidingWindowConvLSTM3D, self).__init__()
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels] * num_layers
        if isinstance(kernel_size, int) or isinstance(kernel_size[0], int):
            kernel_size = [kernel_size] * num_layers

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.station_dim = station_dim
        self.component_dim = component_dim
        self.num_layers = num_layers
        self.window_size = window_size
        self.stride = stride

        cell_list = []
        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_channels[i - 1]
            cell_list.append(
                ConvLSTM3DCell(
                    input_channels=in_ch,
                    hidden_channels=hidden_channels[i],
                    kernel_size=kernel_size[i],
                    station_dim=station_dim,
                    component_dim=component_dim
                )
            )
        self.cell_list = nn.ModuleList(cell_list)

    def _init_hidden(self, batch_size, device):
        hidden = []
        for ch in self.hidden_channels:
            h = torch.zeros(batch_size, ch, self.station_dim, self.component_dim, device=device)
            c = torch.zeros_like(h)
            hidden.append((h, c))
        return hidden

    def forward(self, x, hidden_state=None):
        B, _, T, S, K = x.size()
        device = x.device

        if hidden_state is None:
            hidden_state = self._init_hidden(B, device)

        num_windows = max(1, math.ceil((T - self.window_size) / self.stride) + 1)
        H_last = self.hidden_channels[-1]

        final_output = torch.zeros(B, H_last, T, S, K, device=device)
        count_map = torch.zeros(B, H_last, T, S, K, device=device)

        for w in range(num_windows):
            t0 = w * self.stride
            t1 = min(t0 + self.window_size, T)

            x_win = x[:, :, t0:t1, :, :]
            layer_in = x_win
            current_win_len = layer_in.size(2)

            new_hidden = []

            for i, cell in enumerate(self.cell_list):
                h_prev, c_prev = hidden_state[i]

                h_ext = h_prev.unsqueeze(2).expand(-1, -1, current_win_len, -1, -1)
                c_ext = c_prev.unsqueeze(2).expand(-1, -1, current_win_len, -1, -1)

                h_out, c_out = cell(layer_in, h_ext, c_ext)

                next_state_idx = self.stride - 1
                if next_state_idx >= current_win_len:
                    next_state_idx = current_win_len - 1

                h_new = h_out[:, :, next_state_idx].reshape(B, -1, S, K)
                c_new = c_out[:, :, next_state_idx].reshape(B, -1, S, K)
                new_hidden.append((h_new, c_new))

                layer_in = h_out

            hidden_state = new_hidden

            final_output[:, :, t0:t1, :, :] += layer_in
            count_map[:, :, t0:t1, :, :] += 1.0

        count_map = torch.clamp(count_map, min=1.0)
        final_output = final_output / count_map

        return final_output, hidden_state


class MultiScaleWaveformPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        branch_ch = max(1, input_dim // 8)
        self.multi_scale_convs = nn.ModuleList([
            nn.Conv1d(input_dim, branch_ch, kernel_size=k, padding=k // 2)
            for k in [3, 5, 7, 9]
        ])
        self.fusion_conv = nn.Conv1d(branch_ch * 4, input_dim, kernel_size=1)
        self.wave_predictor = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(input_dim, output_dim, kernel_size=1)
        )

    def forward(self, x):
        x_conv = x.transpose(1, 2)
        feats = [conv(x_conv) for conv in self.multi_scale_convs]
        fused = torch.cat(feats, dim=1)
        fused = self.fusion_conv(fused)
        fused = fused + x_conv
        out = self.wave_predictor(fused)
        return out.transpose(1, 2)


class ImprovedMultiStationConvLSTM3D(nn.Module):
    def __init__(self, input_channels=1, hidden_channels=32, kernel_size=(15, 2, 2),
                 num_input_stations=4, num_output_stations=1, num_components=2,
                 time_steps=6000, window_size=100, stride=10, lstm_hidden=64, lstm_layers=1):
        super().__init__()
        self.num_input_stations = num_input_stations
        self.num_output_stations = num_output_stations
        self.num_components = num_components
        self.hidden_channels = hidden_channels
        self.lstm_hidden = lstm_hidden

        if isinstance(kernel_size, (tuple, list)):
            time_k = kernel_size[0]
            actual_kernel_size = tuple(kernel_size) if len(kernel_size) == 3 else (time_k, 1, num_components)
        else:
            time_k = kernel_size
            if time_k % 2 == 0: time_k += 1
            actual_kernel_size = (time_k, 1, num_components)

        self.convlstm3d = SlidingWindowConvLSTM3D(
            input_channels=input_channels,
            hidden_channels=hidden_channels,
            kernel_size=actual_kernel_size,
            station_dim=num_input_stations,
            component_dim=num_components,
            num_layers=1,
            window_size=window_size,
            stride=stride
        )

        self.ms_convs = nn.ModuleList([
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=(K, 1, 1),
                      padding=(K // 2, 0, 0), bias=True)
            for K in (3, 5)
        ])

        self.fuse = nn.Conv3d(hidden_channels * 2, hidden_channels, kernel_size=1, bias=True)

        self.flattened = hidden_channels * num_input_stations * num_components
        self.bi_lstm = nn.LSTM(
            input_size=self.flattened,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.1 if lstm_layers > 1 else 0.0
        )

        self.amp_encoder = nn.Sequential(
            nn.Linear(num_input_stations * num_components, lstm_hidden),
            nn.LayerNorm(lstm_hidden),
            nn.GELU(),
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.LayerNorm(lstm_hidden),
            nn.GELU()
        )

        self.shared_feature_extractor = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.LayerNorm(lstm_hidden),
            nn.GELU()
        )

        self.wave_predictor = MultiScaleWaveformPredictor(
            input_dim=lstm_hidden,
            output_dim=num_output_stations * num_components
        )

        self.amplitude_network = EnhancedAmplitudeNetwork(
            lstm_feature_size=lstm_hidden * 2,
            amp_feature_size=lstm_hidden,
            output_size=num_output_stations * num_components
        )

        self.base_gate = nn.Sequential(nn.Linear(lstm_hidden * 2, 1), nn.Sigmoid())

    def forward(self, x, input_amps=None):
        B, _, T, S_in, C = x.shape
        device = x.device

        h3d, _ = self.convlstm3d(x)
        feats = torch.cat([conv(h3d) for conv in self.ms_convs], dim=1)
        f = self.fuse(feats)

        seq = f.permute(0, 2, 1, 3, 4).contiguous().view(B, T, -1)
        lstm_out, _ = self.bi_lstm(seq)

        if input_amps is None:
            amp_features = torch.zeros(B, self.lstm_hidden, device=device)
        else:
            flat_amps = input_amps.view(B, 1, -1).squeeze(1)
            amp_features = self.amp_encoder(flat_amps)

        shared_features = self.shared_feature_extractor(lstm_out)
        wave_out = self.wave_predictor(shared_features)
        wave_out = wave_out.view(B, T, self.num_output_stations, self.num_components)

        base_in = x[:, 0, :T, :, :self.num_components]
        base_mean = base_in.mean(dim=2, keepdim=True)
        base = base_mean.expand(B, T, self.num_output_stations, self.num_components)

        gate = self.base_gate(lstm_out.mean(dim=1))
        gate = gate.view(B, 1, 1, 1)

        wave_out = wave_out + 0.05 * gate * base

        amp_out = self.amplitude_network(lstm_out, amp_features)
        amp_out = amp_out.view(B, 1, self.num_output_stations, self.num_components)
        amp_out = amp_out.clamp(min=-12.0, max=6.0)

        wave_out = wave_out - torch.mean(wave_out, dim=1, keepdim=True)

        edge_len = 30
        if T >= edge_len * 2:
            window = 0.5 * (1 - torch.cos(torch.linspace(0, math.pi, edge_len, device=device)))
            wave_out[:, :edge_len, :, :] *= window.view(1, -1, 1, 1) * 0.1
            wave_out[:, -edge_len:, :, :] *= window.flip(0).view(1, -1, 1, 1) * 0.1

        return wave_out, amp_out


# ========================================================
# Part 2: Data loading functions
# ========================================================

def load_seismic_file(file_path):
    """
    [Fix 1] Parse line by line; failed lines are filled with NaN placeholders and linear interpolation to preserve array length and time alignment.
    A single malformed line no longer causes the whole file to return an empty array.
    """
    data = []
    bad_lines = 0

    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Prefer whitespace splitting
            parts = line.split()
            parsed = False

            if len(parts) >= 2:
                try:
                    val = float(parts[1])
                    data.append(val)
                    parsed = True
                except ValueError:
                    pass

            # If whitespace splitting fails, try comma splitting (CSV compatible)
            if not parsed:
                parts_csv = line.split(',')
                if len(parts_csv) >= 2:
                    try:
                        val = float(parts_csv[1])
                        data.append(val)
                        parsed = True
                    except ValueError:
                        pass

            if not parsed:
                bad_lines += 1
                data.append(float('nan'))  # placeholder to preserve array length

        if bad_lines > 0:
            print(f"  Warning: {os.path.basename(file_path)}: {bad_lines}/{len(lines)} lines failed to parse; filled by interpolation")

        arr = np.array(data, dtype=np.float64)

        # Fill NaN values by linear interpolation to preserve time alignment
        nan_mask = np.isnan(arr)
        if nan_mask.any():
            if nan_mask.all():
                print(f"  Error: {os.path.basename(file_path)}: all lines failed to parse; returning a zero array")
                return np.zeros(len(data))
            indices = np.arange(len(arr))
            arr[nan_mask] = np.interp(
                indices[nan_mask],
                indices[~nan_mask],
                arr[~nan_mask]
            )

        return arr

    except Exception as e:
        print(f"  Error: failed to read file {file_path}: {e}")
        return np.array([])


def load_station_data(base_dir, component_dirs, num_samples=None, station_filter=None):
    ext_map = {'x': '.EW.txt', 'y': '.NS.txt', 'px': '.EW.txt', 'py': '.NS.txt'}
    component_name_map = {'x': 'X_component', 'y': 'Y_component', 'px': 'X_component', 'py': 'Y_component'}
    file_mapping = {}

    for comp_dir in component_dirs:
        comp_path = os.path.join(base_dir, comp_dir)
        ext = ext_map.get(comp_dir, '.txt')
        comp_name = component_name_map.get(comp_dir, '')

        if os.path.exists(comp_path):
            station_files = glob.glob(os.path.join(comp_path, f"*{ext}"))
            for file_path in station_files:
                filename = os.path.basename(file_path)
                base_id = filename.replace(ext, '')
                if comp_name and comp_name in base_id:
                    base_id = base_id.replace(f'_{comp_name}', '').replace(f'{comp_name}_', '')
                if base_id not in file_mapping:
                    file_mapping[base_id] = {}
                file_mapping[base_id][comp_dir] = filename

    valid_base_ids = []
    for base_id in file_mapping:
        if all(comp_dir in file_mapping[base_id] for comp_dir in component_dirs):
            valid_base_ids.append(base_id)
    valid_base_ids = sorted(valid_base_ids)

    if station_filter is not None:
        if isinstance(station_filter, int):
            filtered_ids = valid_base_ids[:station_filter]
        elif isinstance(station_filter, list):
            if len(station_filter) > 0 and isinstance(station_filter[0], str):
                filtered_ids = []
                for name in station_filter:
                    matches = [sid for sid in valid_base_ids if name in sid]
                    if matches:
                        filtered_ids.extend(matches)
                filtered_ids = sorted(list(set(filtered_ids)))
            elif len(station_filter) > 0 and isinstance(station_filter[0], int):
                filtered_ids = [valid_base_ids[i] for i in station_filter if i < len(valid_base_ids)]
            else:
                filtered_ids = valid_base_ids
        else:
            filtered_ids = valid_base_ids
    else:
        filtered_ids = valid_base_ids

    data_dict = {base_id: [] for base_id in filtered_ids}
    for comp_dir in component_dirs:
        comp_path = os.path.join(base_dir, comp_dir)
        ext = ext_map.get(comp_dir, '.txt')
        for base_id in filtered_ids:
            if base_id in file_mapping and comp_dir in file_mapping[base_id]:
                filename = file_mapping[base_id][comp_dir]
                file_path = os.path.join(comp_path, filename)
                data = load_seismic_file(file_path)
                if len(data) == 0:
                    print(f"  Warning: failed to load station {base_id}, component {comp_dir}; filling with zeros")
                    data = np.zeros(num_samples or 1000)
                elif num_samples and len(data) > num_samples:
                    data = data[:num_samples]
                data_dict[base_id].append(data)
            else:
                data = np.zeros(num_samples or 1000)
                data_dict[base_id].append(data)

    return data_dict, filtered_ids


def prepare_inference_data(base_dir, input_component_dirs, output_component_dirs=None,
                           time_steps=None, input_station_filter=None, output_station_filter=None):
    input_data, input_station_ids = load_station_data(
        base_dir, input_component_dirs, time_steps, station_filter=input_station_filter
    )
    if not input_data:
        raise ValueError("No input data were loaded")

    output_data, output_station_ids = None, None
    if output_component_dirs:
        output_data, output_station_ids = load_station_data(
            base_dir, output_component_dirs, time_steps, station_filter=output_station_filter
        )

    # [Fix 2] Check component-length consistency to avoid misalignment caused by missing lines in some files
    for station_id in input_data:
        lengths = [len(d) for d in input_data[station_id]]
        if len(set(lengths)) > 1:
            print(f"  Warning: input station {station_id} has inconsistent component lengths: {lengths}; truncating to the shortest")
            min_len = min(lengths)
            input_data[station_id] = [d[:min_len] for d in input_data[station_id]]

    if output_data:
        for station_id in output_data:
            lengths = [len(d) for d in output_data[station_id]]
            if len(set(lengths)) > 1:
                print(f"  Warning: output station {station_id} has inconsistent component lengths: {lengths}; truncating to the shortest")
                min_len = min(lengths)
                output_data[station_id] = [d[:min_len] for d in output_data[station_id]]

    min_length = min(len(d) for station in input_data.values() for d in station)
    if output_data:
        min_length = min(min_length, min(len(d) for station in output_data.values() for d in station))
    if time_steps is not None and min_length > time_steps:
        min_length = time_steps

    num_input_stations = len(input_station_ids)
    num_input_components = len(input_component_dirs)

    X = np.zeros((1, 1, min_length, num_input_stations, num_input_components))
    X_amplitude = np.zeros((1, 1, num_input_stations, num_input_components))

    for i, station_id in enumerate(input_station_ids):
        for j, component_data in enumerate(input_data[station_id]):
            data = component_data[:min_length]
            max_abs = np.max(np.abs(data))
            if max_abs > 0:
                X[0, 0, :, i, j] = data / max_abs
                X_amplitude[0, 0, i, j] = np.log10(max_abs + 1e-8)
            else:
                X[0, 0, :, i, j] = data
                X_amplitude[0, 0, i, j] = np.log10(1e-8)

    X_tensor = torch.FloatTensor(X)
    X_amp_tensor = torch.FloatTensor(X_amplitude)

    if output_data:
        num_output_stations = len(output_station_ids)
        num_output_components = len(output_component_dirs)
        y = np.zeros((1, min_length, num_output_stations, num_output_components))
        y_amplitude = np.zeros((1, 1, num_output_stations, num_output_components))

        for i, station_id in enumerate(output_station_ids):
            for j, component_data in enumerate(output_data[station_id]):
                data = component_data[:min_length]
                max_abs = np.max(np.abs(data))
                if max_abs > 0:
                    y[0, :, i, j] = data / max_abs
                    y_amplitude[0, 0, i, j] = np.log10(max_abs + 1e-8)
                else:
                    y[0, :, i, j] = data
                    y_amplitude[0, 0, i, j] = np.log10(1e-8)

        y_tensor = torch.FloatTensor(y)
        y_amp_tensor = torch.FloatTensor(y_amplitude)
        return X_tensor, X_amp_tensor, y_tensor, y_amp_tensor, input_station_ids, output_station_ids

    return X_tensor, X_amp_tensor, None, None, input_station_ids, None


def prepare_multiple_inference_data(base_dir, input_component_dirs, output_component_dirs=None,
                                    time_steps=None, input_station_filter=None, output_station_filter=None):
    sample_dirs = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    all_inputs, all_input_amps, all_targets, all_target_amps, sample_names = [], [], [], [], []
    all_output_station_names = []
    input_station_names, output_station_names = None, None

    for sample_dir in tqdm(sample_dirs, desc="Loading data", unit="sample", ncols=80):
        sample_path = os.path.join(base_dir, sample_dir)
        try:
            result = prepare_inference_data(
                sample_path, input_component_dirs, output_component_dirs, time_steps,
                input_station_filter=input_station_filter, output_station_filter=output_station_filter
            )
            X, X_amp, y, y_amp, input_ids, output_ids = result
            all_inputs.append(X)
            all_input_amps.append(X_amp)
            sample_names.append(sample_dir)

            if input_station_names is None:
                input_station_names = input_ids

            if output_ids is not None:
                all_output_station_names.append(output_ids)
                if output_station_names is None:
                    output_station_names = output_ids

            if y is not None:
                all_targets.append(y)
                all_target_amps.append(y_amp)

        except Exception as e:
            print(f"Warning: skipping sample {sample_dir}: {e}")
            continue

    if not all_inputs:
        raise ValueError("No sample data were successfully loaded")

    X_all = torch.cat(all_inputs, dim=0)
    X_amp_all = torch.cat(all_input_amps, dim=0)
    y_all = torch.cat(all_targets, dim=0) if all_targets else None
    y_amp_all = torch.cat(all_target_amps, dim=0) if all_target_amps else None

    return X_all, X_amp_all, y_all, y_amp_all, sample_names, input_station_names, output_station_names, all_output_station_names


# ========================================================
# Part 3: Predictor class
# ========================================================

class SeismicPredictor:
    def __init__(self, model_path, model_config, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_config = model_config
        self.model = ImprovedMultiStationConvLSTM3D(**model_config)
        self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)

    def predict(self, input_data, input_amps, batch_size=8):
        self.model.eval()
        total_samples = input_data.shape[0]
        all_pred_waves, all_pred_amps = [], []

        with torch.no_grad():
            for start_idx in tqdm(range(0, total_samples, batch_size), desc="Model prediction", unit="batch", ncols=80):
                end_idx = min(start_idx + batch_size, total_samples)
                batch_input = input_data[start_idx:end_idx].to(self.device)
                batch_amps = input_amps[start_idx:end_idx].to(self.device)
                pred_waves, pred_amps = self.model(batch_input, batch_amps)
                all_pred_waves.append(pred_waves.cpu())
                all_pred_amps.append(pred_amps.cpu())

        return torch.cat(all_pred_waves, dim=0), torch.cat(all_pred_amps, dim=0)


# ========================================================
# Part 4: Visualization and evaluation functions
# ========================================================

def visualize_waveforms(target_data, predicted_data, sample_idx=0, sample_name="unknown",
                        station_names=None, save_dir=None):
    if sample_idx >= target_data.shape[0]:
        return
    target_sample = target_data[sample_idx]
    predicted_sample = predicted_data[sample_idx]
    time_steps, num_stations, num_components = target_sample.shape
    component_names = ['EW', 'NS'][:num_components]
    if station_names is None:
        station_names = [f'Station {i + 1}' for i in range(num_stations)]

    fig, axs = plt.subplots(num_stations, num_components, figsize=(15, 3 * num_stations))
    if num_stations == 1 and num_components == 1:
        axs = np.array([[axs]])
    elif num_stations == 1:
        axs = axs.reshape(1, -1)
    elif num_components == 1:
        axs = axs.reshape(-1, 1)

    for si in range(num_stations):
        for ci in range(num_components):
            ax = axs[si, ci]
            target_wave = target_sample[:, si, ci].cpu().numpy()
            pred_wave = predicted_sample[:, si, ci].cpu().numpy()
            ax.plot(target_wave, 'g-', label='Target', linewidth=1)
            ax.plot(pred_wave, 'r--', label='Prediction', linewidth=1)
            ax.set_title(f'{station_names[si]}, Comp {component_names[ci]}')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)
            if si == 0 and ci == 0:
                ax.legend()

    plt.suptitle(f'Waveform Comparison - {sample_name}')
    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(f'{save_dir}/waveform_{sample_name}.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def visualize_frequency_domain(target, prediction, sampling_rate=100, min_freq=0, max_freq=50,
                               sample_idx=0, title="Spectrum", station_names=None,
                               use_log_scale=False, save_path=None):
    B, T, S, C = target.shape
    component_names = ['EW', 'NS'][:C]
    if station_names is None:
        station_names = [f'Station {i + 1}' for i in range(S)]
    window = np.hanning(T)

    fig, axs = plt.subplots(S, C, figsize=(6 * C, 3 * S), constrained_layout=True)
    if S == 1 and C == 1:
        axs = np.array([[axs]])
    elif S == 1:
        axs = axs.reshape(1, -1)
    elif C == 1:
        axs = axs.reshape(-1, 1)

    for s in range(S):
        for c in range(C):
            ax = axs[s, c]
            target_signal = target[sample_idx, :, s, c].cpu().numpy()
            target_signal = (target_signal - np.mean(target_signal)) * window
            target_fft = np.abs(np.fft.rfft(target_signal))
            pred_signal = prediction[sample_idx, :, s, c].cpu().numpy()
            pred_signal = (pred_signal - np.mean(pred_signal)) * window
            pred_fft = np.abs(np.fft.rfft(pred_signal))
            freq = np.fft.rfftfreq(T, d=1.0 / sampling_rate)
            mask = (freq >= min_freq) & (freq <= max_freq)

            if use_log_scale:
                ax.semilogy(freq[mask], target_fft[mask], 'g-', label='Target', linewidth=1.5)
                ax.semilogy(freq[mask], pred_fft[mask], 'r--', label='Prediction', linewidth=1.5)
            else:
                ax.plot(freq[mask], target_fft[mask], 'g-', label='Target', linewidth=1.5)
                ax.plot(freq[mask], pred_fft[mask], 'r--', label='Prediction', linewidth=1.5)

            ax.set_title(f'{station_names[s]}, Comp {component_names[c]}')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Magnitude')
            ax.grid(True, alpha=0.3)
            ax.legend()

    plt.suptitle(f"{title} ({sample_idx})")
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def evaluate_model_performance(pred_wave, target_wave, pred_amp, target_amp, station_names=None):
    B, T, S, C = pred_wave.shape
    component_names = ['EW', 'NS'][:C]
    if station_names is None:
        station_names = [f'Station{i + 1}' for i in range(S)]
    pred_amp_linear = 10 ** pred_amp
    target_amp_linear = 10 ** target_amp
    pred_orig = pred_wave * pred_amp_linear
    target_orig = target_wave * target_amp_linear
    results = {'station_metrics': {}, 'component_avg': {}, 'overall': {}}

    print("\n" + "=" * 80)
    print("Model performance evaluation - original-amplitude waveforms")
    print("=" * 80)

    for c, comp_name in enumerate(component_names):
        comp_mse_list, comp_rmse_list, comp_mae_list = [], [], []
        for s in tqdm(range(S), desc=f"   Computing {comp_name}", unit="station", ncols=80, leave=False):
            orig_diff = pred_orig[:, :, s, c] - target_orig[:, :, s, c]
            mse = torch.mean(orig_diff ** 2).item()
            rmse = np.sqrt(mse)
            mae = torch.mean(torch.abs(orig_diff)).item()
            key = f"{station_names[s]}_{comp_name}"
            results['station_metrics'][key] = {'mse': mse, 'rmse': rmse, 'mae': mae}
            comp_mse_list.append(mse)
            comp_rmse_list.append(rmse)
            comp_mae_list.append(mae)

        avg_mse = np.mean(comp_mse_list)
        avg_rmse = np.mean(comp_rmse_list)
        avg_mae = np.mean(comp_mae_list)
        results['component_avg'][comp_name] = {'mse': avg_mse, 'rmse': avg_rmse, 'mae': avg_mae}

    overall_mse = torch.mean((pred_orig - target_orig) ** 2).item()
    results['overall'] = {
        'mse': overall_mse,
        'rmse': np.sqrt(overall_mse),
        'mae': torch.mean(torch.abs(pred_orig - target_orig)).item()
    }

    results['ranking'] = {}
    for c, comp_name in enumerate(component_names):
        station_rmse = []
        for s in range(S):
            key = f"{station_names[s]}_{comp_name}"
            rmse = results['station_metrics'][key]['rmse']
            station_rmse.append((station_names[s], rmse))
        results['ranking'][comp_name] = sorted(station_rmse, key=lambda x: x[1], reverse=True)

    return results


def evaluate_normalized_performance(pred_wave, target_wave, station_names=None):
    B, T, S, C = pred_wave.shape
    component_names = ['EW', 'NS'][:C]
    if station_names is None:
        station_names = [f'Station{i + 1}' for i in range(S)]
    results = {'station_metrics': {}, 'component_avg': {}, 'overall': {}}

    print("\n" + "=" * 80)
    print("Model performance evaluation - normalized waveforms")
    print("=" * 80)

    for c, comp_name in enumerate(component_names):
        comp_mse_list, comp_rmse_list, comp_mae_list = [], [], []
        for s in tqdm(range(S), desc=f"   Computing {comp_name}", unit="station", ncols=80, leave=False):
            diff = pred_wave[:, :, s, c] - target_wave[:, :, s, c]
            mse = torch.mean(diff ** 2).item()
            rmse = np.sqrt(mse)
            mae = torch.mean(torch.abs(diff)).item()
            key = f"{station_names[s]}_{comp_name}"
            results['station_metrics'][key] = {'mse': mse, 'rmse': rmse, 'mae': mae}
            comp_mse_list.append(mse)
            comp_rmse_list.append(rmse)
            comp_mae_list.append(mae)

        avg_mse = np.mean(comp_mse_list)
        avg_rmse = np.mean(comp_rmse_list)
        avg_mae = np.mean(comp_mae_list)
        results['component_avg'][comp_name] = {'mse': avg_mse, 'rmse': avg_rmse, 'mae': avg_mae}

    overall_mse = torch.mean((pred_wave - target_wave) ** 2).item()
    results['overall'] = {
        'mse': overall_mse,
        'rmse': np.sqrt(overall_mse),
        'mae': torch.mean(torch.abs(pred_wave - target_wave)).item()
    }

    results['ranking'] = {}
    for c, comp_name in enumerate(component_names):
        station_rmse = []
        for s in range(S):
            key = f"{station_names[s]}_{comp_name}"
            rmse = results['station_metrics'][key]['rmse']
            station_rmse.append((station_names[s], rmse))
        results['ranking'][comp_name] = sorted(station_rmse, key=lambda x: x[1], reverse=True)

    return results


# ========================================================
# Parallel save task function
# ========================================================

def save_single_sample_job(i, pred_wave, target_wave, pred_amp, target_amp,
                           station_names, component_names, dt, output_dir, save_original):
    """
    Parallel task: save all waveform files for a single sample.
    [Fix 3] Time-column format changed to %.4f (fixed-point) to avoid repeated identical time values (jitter) caused by insufficient precision of scientific notation.
    """
    T, S, C = pred_wave.shape
    # Time axis: use a high-precision floating-point array to avoid accumulated error
    time_axis = np.arange(T, dtype=np.float64) * dt

    sample_dir = os.path.join(output_dir, f"sample_{i:03d}")
    os.makedirs(sample_dir, exist_ok=True)

    if save_original:
        p_amp_factor = (10 ** pred_amp).squeeze(0)
        t_amp_factor = (10 ** target_amp).squeeze(0)

    count = 0

    for s in range(S):
        st_name = station_names[s]
        for c in range(C):
            comp_name = component_names[c]

            p_data = pred_wave[:, s, c].astype(np.float64)
            t_data = target_wave[:, s, c].astype(np.float64)

            if save_original:
                p_data = p_data * p_amp_factor[s, c]
                t_data = t_data * t_amp_factor[s, c]

            # [Fix 3 core] Time column uses %.4f (fixed-point); data columns use %.6e (scientific notation)
            # This ensures that a 0.01 s time step remains exact at 250 s and avoids many identical time values
            data_to_save = np.column_stack((time_axis, p_data, t_data))
            filename = f"wave_{st_name}_{comp_name}.csv"
            filepath = os.path.join(sample_dir, filename)

            np.savetxt(
                filepath,
                data_to_save,
                fmt=["%.4f", "%.6e", "%.6e"],   # fixed-point format for the time column
                delimiter=",",
                header="Time,Predicted,Target",
                comments=""
            )
            count += 1

    return count


def save_waveform_records(pred_waves, target_waves, pred_amps, target_amps,
                          output_dir="./waveform_records", sample_indices=None,
                          station_names=None, dt=0.01, save_original=True,
                          per_sample_station_names=None):
    os.makedirs(output_dir, exist_ok=True)
    B, T, S, C = pred_waves.shape
    component_names = ['EW', 'NS'][:C]

    if station_names is None:
        station_names = [f'station{i + 1}' for i in range(S)]

    if sample_indices is None:
        sample_indices = list(range(min(20, B)))
    else:
        sample_indices = list(sample_indices)

    valid_indices = [i for i in sample_indices if i < B]
    if not valid_indices:
        return 0

    print(f"   - Preparing to save waveform data for {len(valid_indices)} samples...")

    try:
        pw_np = pred_waves.detach().cpu().numpy()
        tw_np = target_waves.detach().cpu().numpy()
        pa_np = pred_amps.detach().cpu().numpy()
        ta_np = target_amps.detach().cpu().numpy()
        n_jobs = min(16, os.cpu_count())
    except Exception as e:
        print(f"Warning: data conversion failed; switching to single-core execution: {e}")
        pw_np, tw_np, pa_np, ta_np = pred_waves, target_waves, pred_amps, target_amps
        n_jobs = 1

    if len(valid_indices) < 5:
        n_jobs = 1

    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(save_single_sample_job)(
            i,
            pw_np[i],
            tw_np[i],
            pa_np[i],
            ta_np[i],
            per_sample_station_names[i] if (
                        per_sample_station_names is not None and i < len(per_sample_station_names)) else station_names,
            component_names,
            dt,
            output_dir,
            save_original
        )
        for i in tqdm(valid_indices, desc="Saving files", unit="sample", ncols=80)
    )

    total_files = sum(results)
    print(f"Saved {total_files} waveform files to: {output_dir}")
    return total_files


# ========================================================
# Part 5: Main function
# ========================================================

def main():
    print("=" * 60)
    print("Seismic waveform prediction model - batch inference program (multi-process accelerated version)")
    print("=" * 60)

    # 1. Core path configuration
    models_dir = r"E:\fault_vs_no_fault_10runs\chapter3_single_stage_training"
    output_root_dir = r"E:\fault_vs_no_fault_10runs\chapter3_single_stage_training"
    test_data_dir = r"C:\Users\Lenovo\Desktop\test_set - copy (2)"

    # 2. Model configuration
    model_config = {
        'input_channels': 1,
        'hidden_channels': 16,
        'kernel_size': (15, 1, 2),
        'num_input_stations': 4,
        'num_output_stations': 30,
        'num_components': 2,
        'time_steps': 25000,
        'window_size': 3000,
        'stride': 2000,
        'lstm_hidden': 24,
        'lstm_layers': 1
    }

    # Sampling rate (Hz), which determines the time-column precision; modify according to the actual data
    SAMPLING_RATE = 100
    dt = 1.0 / SAMPLING_RATE

    input_component_dirs = ['x', 'y']
    output_component_dirs = ['px', 'py']
    time_steps = 25000
    INPUT_STATION_FILTER = ['ALP', 'MOP', 'PLS', 'RIO']
    OUTPUT_STATION_FILTER = None

    save_waveform_data = True
    save_visualizations = True
    num_visualize_samples = 9999999

    # 3. Scan models
    if not os.path.exists(models_dir):
        print(f"Model folder does not exist: {models_dir}")
        return
    model_files = glob.glob(os.path.join(models_dir, "*.pth"))
    model_files.sort()
    if not model_files:
        print(f"No .pth model files were found")
        return
    print(f"\nFound {len(model_files)} models to process")

    # 4. Load data
    print("\n" + "=" * 60)
    print(f"Loading test data (shared by all models)...")
    print("=" * 60)
    try:
        (X_test, X_amp_test, y_test, y_amp_test, sample_names,
         input_station_names, output_station_names, all_output_station_names) = \
            prepare_multiple_inference_data(
                test_data_dir, input_component_dirs, output_component_dirs, time_steps,
                input_station_filter=INPUT_STATION_FILTER,
                output_station_filter=OUTPUT_STATION_FILTER
            )

        if X_test.shape[3] != model_config['num_input_stations']:
            print(f"Error: input-station count mismatch! Expected {model_config['num_input_stations']}, got {X_test.shape[3]}")
            return
        print(f"Data loading completed: {len(sample_names)} samples")
        print(f"   Input stations: {input_station_names}")
        print(f"   Number of output stations: {len(output_station_names) if output_station_names else 0}")
    except Exception as e:
        print(f"Error: failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return

    # 5. Loop over each model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Computing device: {device}")

    for idx, model_path in enumerate(model_files):
        model_filename = os.path.basename(model_path)
        model_name_pure = os.path.splitext(model_filename)[0]

        print("\n" + "#" * 80)
        print(f"[{idx + 1}/{len(model_files)}] Processing model: {model_filename}")
        print("#" * 80)

        current_output_dir = os.path.join(output_root_dir, model_name_pure)
        os.makedirs(current_output_dir, exist_ok=True)

        try:
            predictor = SeismicPredictor(model_path, model_config, device)
        except Exception as e:
            print(f"Error: failed to load: {e}")
            continue

        print("Starting inference...")
        pred_waves, pred_amps = predictor.predict(X_test, X_amp_test, batch_size=8)

        metrics, metrics_norm = None, None
        if y_test is not None:
            print("Computing metrics...")
            metrics = evaluate_model_performance(pred_waves, y_test, pred_amps, y_amp_test,
                                                 station_names=output_station_names)
            metrics_norm = evaluate_normalized_performance(pred_waves, y_test,
                                                           station_names=output_station_names)

        viz_dir = os.path.join(current_output_dir, "visualizations")
        wave_dir = os.path.join(current_output_dir, "waveform_data")

        if y_test is not None:
            pred_amp_linear = 10 ** pred_amps
            target_amp_linear = 10 ** y_amp_test
            pred_orig = pred_waves * pred_amp_linear
            target_orig = y_test * target_amp_linear

            if save_visualizations:
                print("Generating figures...")
                sample_indices = list(range(min(num_visualize_samples, len(sample_names))))
                n_jobs = max(1, os.cpu_count() // 2)
                Parallel(n_jobs=n_jobs, backend='loky')(
                    delayed(generate_single_visualization)(
                        i, target_orig[i], pred_orig[i], sample_names[i], output_station_names, viz_dir
                    ) for i in tqdm(sample_indices, desc="Plotting progress", unit="figure", ncols=80)
                )

            if save_waveform_data:
                print("Saving data (multi-process accelerated)...")
                save_indices = list(range(min(num_visualize_samples, len(sample_names))))
                save_waveform_records(
                    pred_waves, y_test, pred_amps, y_amp_test,
                    output_dir=wave_dir,
                    sample_indices=save_indices,
                    station_names=output_station_names,
                    dt=dt,                                        # [Fix 3] explicitly pass the correct dt
                    save_original=True,
                    per_sample_station_names=all_output_station_names
                )

        # Generate report
        report_path = os.path.join(current_output_dir, "inference_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Model: {model_filename}\nTime: {datetime.now()}\n")
            f.write(f"Sampling rate: {SAMPLING_RATE} Hz, dt: {dt:.4f} s\n")
            f.write(f"Number of samples: {len(sample_names)}\n")
            if metrics:
                f.write(
                    f"\n[Original waveforms]\n"
                    f"MSE:  {metrics['overall']['mse']:.6f}\n"
                    f"RMSE: {metrics['overall']['rmse']:.6f}\n"
                    f"MAE:  {metrics['overall']['mae']:.6f}\n"
                )
                f.write("\n[Component averages]\n")
                for comp, vals in metrics['component_avg'].items():
                    f.write(f"  {comp} - MSE: {vals['mse']:.6f}, RMSE: {vals['rmse']:.6f}, MAE: {vals['mae']:.6f}\n")
            if metrics_norm:
                f.write(
                    f"\n[Normalized waveforms]\n"
                    f"MSE:  {metrics_norm['overall']['mse']:.6f}\n"
                    f"RMSE: {metrics_norm['overall']['rmse']:.6f}\n"
                    f"MAE:  {metrics_norm['overall']['mae']:.6f}\n"
                )
                f.write("\n[Component averages]\n")
                for comp, vals in metrics_norm['component_avg'].items():
                    f.write(f"  {comp} - MSE: {vals['mse']:.6f}, RMSE: {vals['rmse']:.6f}, MAE: {vals['mae']:.6f}\n")
        print(f"Report saved: {report_path}")

        del predictor, pred_waves, pred_amps, metrics, metrics_norm
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print(f"All done! Results directory: {output_root_dir}")


if __name__ == "__main__":
    main()