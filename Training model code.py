# ========================================================
# Seismic waveform prediction - baseline model (no fault embedding) - single-stage version
# For comparison with the FaultConditionedModel
# The architecture is identical to the fault-conditioned version, except that the fault embedding and four injection points are removed
# Modification: changed from multi-stage training to single-stage training + loss-proportion monitoring
# ========================================================

import os
import glob
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from datetime import datetime
import json
from pathlib import Path
import warnings
import concurrent.futures
import platform
from matplotlib import font_manager


def set_matplotlib_font():
    candidates = [
        "SimHei", "Microsoft YaHei", "SimSun", "PingFang SC",
        "Noto Sans CJK SC", "WenQuanYi Zen Hei", "DejaVu Sans"
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    chosen = next((n for n in candidates if n in available), "DejaVu Sans")
    plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False
    warnings.filterwarnings("ignore", message="findfont: Font family")

set_matplotlib_font()


# ========================================================
# Part 1: Data loading (identical to the fault-conditioned version)
# ========================================================
def load_seismic_file(file_path):
    try:
        return np.loadtxt(file_path, usecols=1, comments='#')
    except Exception:
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            data = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    data.append(float(parts[1]))
            return np.array(data)
        except Exception:
            return np.array([])


def load_station_data(base_dir, component_dirs, num_samples=None, station_filter=None):
    ext_map = {'x': '.EW.txt', 'y': '.NS.txt', 'px': '.EW.txt', 'py': '.NS.txt'}
    component_name_map = {'x': 'X_component', 'y': 'Y_component',
                          'px': 'X_component', 'py': 'Y_component'}
    file_mapping = {}
    for comp_dir in component_dirs:
        comp_path = os.path.join(base_dir, comp_dir)
        ext = ext_map.get(comp_dir, '.txt')
        comp_name = component_name_map.get(comp_dir, '')
        if os.path.exists(comp_path):
            for file_path in glob.glob(os.path.join(comp_path, f"*{ext}")):
                filename = os.path.basename(file_path)
                base_id = filename.replace(ext, '')
                if comp_name and comp_name in base_id:
                    base_id = base_id.replace(f'_{comp_name}', '').replace(f'{comp_name}_', '')
                file_mapping.setdefault(base_id, {})[comp_dir] = filename

    valid_base_ids = sorted([
        bid for bid in file_mapping
        if all(cd in file_mapping[bid] for cd in component_dirs)
    ])

    if station_filter is None:
        filtered_ids = valid_base_ids
    elif isinstance(station_filter, int):
        filtered_ids = valid_base_ids[:station_filter]
    elif isinstance(station_filter, list):
        if station_filter and isinstance(station_filter[0], str):
            filtered_ids = sorted(set(
                sid for name in station_filter
                for sid in valid_base_ids if name in sid
            ))
        elif station_filter and isinstance(station_filter[0], int):
            filtered_ids = [valid_base_ids[i] for i in station_filter if i < len(valid_base_ids)]
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
                file_path = os.path.join(comp_path, file_mapping[base_id][comp_dir])
                data = load_seismic_file(file_path)
                if len(data) == 0:
                    data = np.zeros(num_samples or 1000)
                elif num_samples and len(data) > num_samples:
                    data = data[:num_samples]
            else:
                data = np.zeros(num_samples or 1000)
            data_dict[base_id].append(data)
    return data_dict, filtered_ids


def prepare_dataset_improved(base_dir, input_component_dirs, output_component_dirs,
                              time_steps=None, input_station_filter=None, output_station_filter=None):
    input_data, input_station_ids = load_station_data(
        base_dir, input_component_dirs, time_steps, station_filter=input_station_filter)
    output_data, output_station_ids = load_station_data(
        base_dir, output_component_dirs, time_steps, station_filter=output_station_filter)

    if not input_data or not output_data:
        return None
    try:
        min_length = min(
            min(len(d) for s in input_data.values() for d in s),
            min(len(d) for s in output_data.values() for d in s))
    except ValueError:
        return None
    if min_length == 0:
        return None
    if time_steps is not None and min_length > time_steps:
        min_length = time_steps

    num_in = len(input_station_ids)
    num_out = len(output_station_ids)
    num_ic = len(input_component_dirs)
    if num_in == 0 or num_out == 0:
        return None

    X = np.zeros((1, 1, min_length, num_in, num_ic))
    X_amp = np.zeros((1, 1, num_in, num_ic))
    y = np.zeros((1, min_length, num_out, len(output_component_dirs)))
    y_amp = np.zeros((1, 1, num_out, len(output_component_dirs)))

    for i, sid in enumerate(input_station_ids):
        for j, cdata in enumerate(input_data[sid]):
            data = cdata[:min_length]
            mx = np.max(np.abs(data))
            if mx > 0:
                X[0, 0, :, i, j] = data / mx
                X_amp[0, 0, i, j] = np.log10(mx + 1e-8)
            else:
                X_amp[0, 0, i, j] = np.log10(1e-8)

    for i, sid in enumerate(output_station_ids):
        for j, cdata in enumerate(output_data[sid]):
            data = cdata[:min_length]
            mx = np.max(np.abs(data))
            if mx > 0:
                y[0, :, i, j] = data / mx
                y_amp[0, 0, i, j] = np.log10(mx + 1e-8)
            else:
                y_amp[0, 0, i, j] = np.log10(1e-8)

    return (torch.FloatTensor(X), torch.FloatTensor(X_amp),
            torch.FloatTensor(y), torch.FloatTensor(y_amp),
            input_station_ids, output_station_ids)


# ========================================================
# Part 2: Dataset class (no fault_id)
# ========================================================
class BaselineDataset(Dataset):
    def __init__(self, inputs, input_amps, targets, target_amps):
        self.inputs = inputs          # (N, 1, T, S_in, C)
        self.input_amps = input_amps  # (N, 1, S_in, C)
        self.targets = targets        # (N, T, S_out, C)
        self.target_amps = target_amps

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (self.inputs[idx], self.input_amps[idx],
                self.targets[idx], self.target_amps[idx])


def _process_single_sample(args):
    sample_dir, input_dirs, output_dirs, t_steps, in_filter, out_filter = args
    try:
        result = prepare_dataset_improved(
            sample_dir, input_dirs, output_dirs, t_steps, in_filter, out_filter)
        if result is None:
            return None
        return result
    except Exception:
        return None


def build_baseline_dataset(data_dir, input_component_dirs, output_component_dirs,
                            input_station_filter, output_station_filter,
                            time_steps, verbose=True, use_multiprocess=True):
    """Scan the directory and load all samples directly; no classifier is required."""
    sample_dirs = []
    for root, dirs, _ in os.walk(data_dir):
        if 'x' in dirs and 'y' in dirs:
            sample_dirs.append(root)
    sample_dirs = sorted(sample_dirs)

    if verbose:
        print(f"   Found {len(sample_dirs)} samples")

    all_X, all_X_amp, all_y, all_y_amp = [], [], [], []
    in_names, out_names = None, None

    use_parallel = (use_multiprocess
                    and len(sample_dirs) > 150
                    and platform.system() != 'Windows')

    if use_parallel:
        num_workers = min(8, os.cpu_count() or 4)
        if verbose:
            print(f"Multi-process loading (workers: {num_workers}) ...")
        task_args = [
            (sd, input_component_dirs, output_component_dirs,
             time_steps, input_station_filter, output_station_filter)
            for sd in sample_dirs
        ]
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(
                executor.map(_process_single_sample, task_args),
                total=len(task_args), desc="Loading waveforms (multi-process)", unit="sample"
            ) if verbose else executor.map(_process_single_sample, task_args))
    else:
        if verbose:
            print(f"Single-process loading...")
        results = []
        iterator = tqdm(sample_dirs, desc="Loading waveforms", leave=False) if verbose else sample_dirs
        for sd in iterator:
            results.append(_process_single_sample(
                (sd, input_component_dirs, output_component_dirs,
                 time_steps, input_station_filter, output_station_filter)
            ))

    for res in results:
        if res is None:
            continue
        X, X_amp, y, y_amp, in_ids, out_ids = res
        all_X.append(X)
        all_X_amp.append(X_amp)
        all_y.append(y)
        all_y_amp.append(y_amp)
        if in_names is None:
            in_names = in_ids
        if out_names is None:
            out_names = out_ids

    if not all_X:
        print("No valid waveform data!")
        return None, None, None

    X_cat     = torch.cat(all_X, dim=0)
    X_amp_cat = torch.cat(all_X_amp, dim=0)
    y_cat     = torch.cat(all_y, dim=0)
    y_amp_cat = torch.cat(all_y_amp, dim=0)

    if verbose:
        print(f"   Data shapes: X={X_cat.shape}, y={y_cat.shape}")
        print(f"   Input stations: {in_names}")
        print(f"   Output stations: {out_names}")

    dataset = BaselineDataset(X_cat, X_amp_cat, y_cat, y_amp_cat)
    return dataset, in_names, out_names


# ========================================================
# Part 3: Model components (identical to the fault-conditioned version)
# ========================================================
class TemporalAttentionPool(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, 1))

    def forward(self, x):
        w = torch.softmax(self.attention(x), dim=1)
        return torch.sum(x * w, dim=1)


class EnhancedAmplitudeNetwork(nn.Module):
    def __init__(self, lstm_feature_size, amp_feature_size, output_size, hidden_size=128):
        super().__init__()
        self.attention_extractor = TemporalAttentionPool(lstm_feature_size)
        self.fusion_network = nn.Sequential(
            nn.Linear(lstm_feature_size + amp_feature_size, hidden_size),
            nn.LayerNorm(hidden_size), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size), nn.GELU(),
            nn.Linear(hidden_size, output_size))
        self.scaling = nn.Parameter(torch.ones(1, output_size) * 1.5)
        self.bias    = nn.Parameter(torch.zeros(1, output_size))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, lstm_features, amp_features):
        combined = torch.cat([self.attention_extractor(lstm_features), amp_features], dim=1)
        return self.fusion_network(combined) * self.scaling + self.bias


class ConvLSTM3DCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, station_dim, component_dim):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        self.hidden_channels = hidden_channels
        self.Wxi = nn.Conv3d(input_channels,  hidden_channels, kernel_size, padding='same', bias=True)
        self.Whi = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding='same', bias=True)
        self.Wxf = nn.Conv3d(input_channels,  hidden_channels, kernel_size, padding='same', bias=True)
        self.Whf = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding='same', bias=True)
        self.Wxc = nn.Conv3d(input_channels,  hidden_channels, kernel_size, padding='same', bias=True)
        self.Whc = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding='same', bias=True)
        self.Wxo = nn.Conv3d(input_channels,  hidden_channels, kernel_size, padding='same', bias=True)
        self.Who = nn.Conv3d(hidden_channels, hidden_channels, kernel_size, padding='same', bias=True)
        self.Wci = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1, 1))
        self.Wcf = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1, 1))
        self.Wco = nn.Parameter(torch.zeros(1, hidden_channels, 1, 1, 1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if self.Wxf.bias is not None: nn.init.constant_(self.Wxf.bias, 1.0)
        if self.Whf.bias is not None: nn.init.constant_(self.Whf.bias, 1.0)
        nn.init.zeros_(self.Wci); nn.init.zeros_(self.Wcf); nn.init.zeros_(self.Wco)

    def forward(self, x, h, c):
        i  = torch.sigmoid(self.Wxi(x) + self.Whi(h) + c * self.Wci)
        f  = torch.sigmoid(self.Wxf(x) + self.Whf(h) + c * self.Wcf)
        g  = torch.tanh(self.Wxc(x) + self.Whc(h))
        cc = f * c + i * g
        o  = torch.sigmoid(self.Wxo(x) + self.Who(h) + cc * self.Wco)
        return o * torch.tanh(cc), cc


class SlidingWindowConvLSTM3D(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size,
                 station_dim, component_dim, num_layers=1, window_size=50, stride=10):
        super().__init__()
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels] * num_layers
        if isinstance(kernel_size, int) or isinstance(kernel_size[0], int):
            kernel_size = [kernel_size] * num_layers
        self.hidden_channels = hidden_channels
        self.station_dim     = station_dim
        self.component_dim   = component_dim
        self.num_layers      = num_layers
        self.window_size     = window_size
        self.stride          = stride
        self.cell_list = nn.ModuleList([
            ConvLSTM3DCell(
                input_channels if i == 0 else hidden_channels[i-1],
                hidden_channels[i], kernel_size[i], station_dim, component_dim
            ) for i in range(num_layers)
        ])

    def _init_hidden(self, B, device):
        return [(torch.zeros(B, ch, self.station_dim, self.component_dim, device=device),
                 torch.zeros(B, ch, self.station_dim, self.component_dim, device=device))
                for ch in self.hidden_channels]

    def forward(self, x, hidden_state=None):
        B, _, T, S, K = x.size()
        device = x.device
        if hidden_state is None:
            hidden_state = self._init_hidden(B, device)
        num_windows  = max(1, math.ceil((T - self.window_size) / self.stride) + 1)
        H_last       = self.hidden_channels[-1]
        final_output = torch.zeros(B, H_last, T, S, K, device=device)
        count_map    = torch.zeros(B, H_last, T, S, K, device=device)

        for w in range(num_windows):
            t0, t1   = w * self.stride, min(w * self.stride + self.window_size, T)
            layer_in = x[:, :, t0:t1, :, :]
            win_len  = layer_in.size(2)
            new_hidden = []
            for i, cell in enumerate(self.cell_list):
                h_prev, c_prev = hidden_state[i]
                h_out, c_out = cell(
                    layer_in,
                    h_prev.unsqueeze(2).expand(-1, -1, win_len, -1, -1),
                    c_prev.unsqueeze(2).expand(-1, -1, win_len, -1, -1))
                idx = min(self.stride - 1, win_len - 1)
                new_hidden.append((h_out[:, :, idx].view(B, -1, S, K),
                                   c_out[:, :, idx].view(B, -1, S, K)))
                layer_in = h_out
            hidden_state = new_hidden
            final_output[:, :, t0:t1] += layer_in
            count_map[:, :, t0:t1]    += 1.0

        return final_output / torch.clamp(count_map, min=1.0), hidden_state


class MultiScaleWaveformPredictor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        branch_ch = max(1, input_dim // 8)
        self.multi_scale_convs = nn.ModuleList([
            nn.Conv1d(input_dim, branch_ch, kernel_size=k, padding=k // 2)
            for k in [3, 5, 7, 9]])
        self.fusion_conv  = nn.Conv1d(branch_ch * 4, input_dim, kernel_size=1)
        self.wave_predictor = nn.Sequential(
            nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1), nn.GELU(),
            nn.Conv1d(input_dim, output_dim, kernel_size=1))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        xc    = x.transpose(1, 2)
        fused = self.fusion_conv(torch.cat([c(xc) for c in self.multi_scale_convs], dim=1))
        return self.wave_predictor(fused + xc).transpose(1, 2)


# ========================================================
# Part 4: Baseline model (no fault embedding; all other components identical)
# ========================================================
class BaselineModel(nn.Module):
    """
    The architecture is identical to the FaultConditionedModel,
    except that the fault embedding and four injection points are removed, serving as a comparison baseline.
    """
    def __init__(self, input_channels=1, hidden_channels=32, kernel_size=(15, 1, 1),
                 num_input_stations=2, num_output_stations=4, num_components=2,
                 time_steps=6000, window_size=100, stride=10,
                 lstm_hidden=64, lstm_layers=1):
        super().__init__()

        self.num_input_stations  = num_input_stations
        self.num_output_stations = num_output_stations
        self.num_components      = num_components
        self.hidden_channels     = hidden_channels
        self.lstm_hidden         = lstm_hidden

        if isinstance(kernel_size, (tuple, list)) and len(kernel_size) == 3:
            time_k, station_k, comp_k = kernel_size
        else:
            time_k     = kernel_size[0] if isinstance(kernel_size, (tuple, list)) else kernel_size
            station_k  = num_input_stations
            comp_k     = num_components
        if time_k % 2 == 0:
            time_k += 1
        actual_kernel_size = (time_k, station_k, comp_k)
        print(f"[BaselineModel] Kernel: {actual_kernel_size}")

        self.flattened = hidden_channels * num_input_stations * num_components

        self.convlstm3d = SlidingWindowConvLSTM3D(
            input_channels, hidden_channels, actual_kernel_size,
            num_input_stations, num_components,
            num_layers=1, window_size=window_size, stride=stride)

        self.ms_convs = nn.ModuleList([
            nn.Conv3d(hidden_channels, hidden_channels, kernel_size=(K, 1, 1),
                      padding=(K // 2, 0, 0)) for K in (3, 5)])
        self.fuse = nn.Conv3d(hidden_channels * 2, hidden_channels, kernel_size=1)

        self.bi_lstm = nn.LSTM(
            input_size=self.flattened, hidden_size=lstm_hidden,
            num_layers=lstm_layers, batch_first=True, bidirectional=True,
            dropout=0.1 if lstm_layers > 1 else 0.0)

        self.amp_encoder = nn.Sequential(
            nn.Linear(num_input_stations * num_components, lstm_hidden),
            nn.LayerNorm(lstm_hidden), nn.GELU(),
            nn.Linear(lstm_hidden, lstm_hidden),
            nn.LayerNorm(lstm_hidden), nn.GELU())

        self.shared_feature_extractor = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.LayerNorm(lstm_hidden), nn.GELU())

        self.wave_predictor = MultiScaleWaveformPredictor(
            lstm_hidden, num_output_stations * num_components)

        self.amplitude_network = EnhancedAmplitudeNetwork(
            lstm_hidden * 2, lstm_hidden, num_output_stations * num_components)

        self.base_gate = nn.Sequential(nn.Linear(lstm_hidden * 2, 1), nn.Sigmoid())

    def forward(self, x, input_amps):
        B, _, T, S_in, C = x.shape
        device = x.device

        h3d, _ = self.convlstm3d(x)
        feats  = torch.cat([conv(h3d) for conv in self.ms_convs], dim=1)
        f      = self.fuse(feats)
        seq    = f.permute(0, 2, 1, 3, 4).contiguous().view(B, T, -1)

        lstm_out, _ = self.bi_lstm(seq)

        flat_amps   = input_amps.view(B, -1)
        amp_features = self.amp_encoder(flat_amps)

        shared   = self.shared_feature_extractor(lstm_out)
        wave_out = self.wave_predictor(shared)
        wave_out = wave_out.view(B, T, self.num_output_stations, self.num_components)

        base_in  = x[:, 0, :T, :, :self.num_components]
        base_mean = base_in.mean(dim=2, keepdim=True)
        base     = base_mean.expand(B, T, self.num_output_stations, self.num_components)
        gate     = self.base_gate(lstm_out.mean(dim=1)).view(B, 1, 1, 1)
        wave_out = wave_out + 0.05 * gate * base

        amp_out = self.amplitude_network(lstm_out, amp_features)
        amp_out = amp_out.view(B, 1, self.num_output_stations, self.num_components)
        amp_out = amp_out.clamp(min=-12.0, max=6.0)

        wave_out = wave_out - torch.mean(wave_out, dim=1, keepdim=True)
        edge_len = 30
        if T >= edge_len * 2:
            w = 0.5 * (1 - torch.cos(torch.linspace(0, math.pi, edge_len, device=device)))
            wave_out[:, :edge_len]  *= w.view(1, -1, 1, 1) * 0.1
            wave_out[:, -edge_len:] *= w.flip(0).view(1, -1, 1, 1) * 0.1

        return wave_out, amp_out


# ========================================================
# Part 5: Loss function (single-stage version - fixed weights)
# ========================================================
class SimpleDualOutputLoss(nn.Module):
    def __init__(self, w_wave=1.0, w_amp=0.1, w_spec=0.1, huber_delta=1.0):
        super().__init__()
        self.huber = nn.HuberLoss(reduction='mean', delta=huber_delta)
        self.w_wave = w_wave
        self.w_amp = w_amp
        self.w_spec = w_spec

    def _spectral_loss(self, pred, target):
        return self.huber(torch.abs(torch.fft.rfft(pred, dim=1)),
                          torch.abs(torch.fft.rfft(target, dim=1)))

    def forward(self, pred_waves, pred_amps, target_waves, target_amps):
        wave_loss = self.huber(pred_waves, target_waves)
        amp_loss = self.huber(pred_amps, target_amps)
        spec_loss = self._spectral_loss(pred_waves, target_waves)
        total = self.w_wave * wave_loss + self.w_amp * amp_loss + self.w_spec * spec_loss
        return total, wave_loss, amp_loss, spec_loss


# ========================================================
# Part 6: Training function (single-stage version + loss-proportion monitoring)
# ========================================================
def train_model(model, train_loader, val_loader, config, device,
                run_id=1, save_dir='.', verbose=True):
    model.to(device)

    w_wave = config.get('w_wave', 1.0)
    w_amp = config.get('w_amp', 0.1)
    w_spec = config.get('w_spec', 0.1)
    criterion = SimpleDualOutputLoss(w_wave=w_wave, w_amp=w_amp, w_spec=w_spec)

    lr = config.get('lr', 1e-3)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    total_epochs = config.get('total_epochs', 40)
    accum = config.get('accumulation_steps', 2)
    patience = config.get('patience', 80)

    steps_per_epoch = math.ceil(len(train_loader) / accum)
    total_steps = max(1, total_epochs * steps_per_epoch)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=total_steps,
        div_factor=25, pct_start=0.3, final_div_factor=1000)

    best_val_loss = float('inf')
    best_state = None
    no_improve_count = 0

    history = {
        'train_total': [], 'train_wave': [], 'train_amp': [], 'train_spec': [],
        'val_total': [],   'val_wave': [],   'val_amp': [],   'val_spec': [],
        'train_wave_contrib': [], 'train_amp_contrib': [], 'train_spec_contrib': [],
        'val_wave_contrib': [],   'val_amp_contrib': [],   'val_spec_contrib': [],
    }

    for epoch in range(total_epochs):
        # ---------- Train ----------
        model.train()
        epoch_loss = {'total': 0., 'wave': 0., 'amp': 0., 'spec': 0.}
        optimizer.zero_grad()

        train_iter = tqdm(train_loader, desc=f"Run{run_id} E{epoch+1}/{total_epochs}",
                          leave=False) if verbose else train_loader

        for batch_idx, (inputs, in_amps, targets, tgt_amps) in enumerate(train_iter):
            inputs   = inputs.to(device)
            in_amps  = in_amps.to(device)
            targets  = targets.to(device)
            tgt_amps = tgt_amps.to(device)

            with torch.cuda.amp.autocast(enabled=scaler is not None):
                pw, pa = model(inputs, in_amps)
                loss, w_loss, a_loss, s_loss = criterion(pw, pa, targets, tgt_amps)
                scaled_loss = loss / accum

            if scaler:
                scaler.scale(scaled_loss).backward()
                if (batch_idx + 1) % accum == 0 or (batch_idx + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                scaled_loss.backward()
                if (batch_idx + 1) % accum == 0 or (batch_idx + 1) == len(train_loader):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            epoch_loss['total'] += loss.item()
            epoch_loss['wave']  += w_loss.item()
            epoch_loss['amp']   += a_loss.item()
            epoch_loss['spec']  += s_loss.item()

        n_batches = max(len(train_loader), 1)
        for k in epoch_loss:
            epoch_loss[k] /= n_batches

        history['train_total'].append(epoch_loss['total'])
        history['train_wave'].append(epoch_loss['wave'])
        history['train_amp'].append(epoch_loss['amp'])
        history['train_spec'].append(epoch_loss['spec'])

        wave_c = w_wave * epoch_loss['wave']
        amp_c  = w_amp  * epoch_loss['amp']
        spec_c = w_spec * epoch_loss['spec']
        history['train_wave_contrib'].append(wave_c)
        history['train_amp_contrib'].append(amp_c)
        history['train_spec_contrib'].append(spec_c)

        # ---------- Validation ----------
        model.eval()
        val_loss = {'total': 0., 'wave': 0., 'amp': 0., 'spec': 0.}
        with torch.no_grad():
            for inputs, in_amps, targets, tgt_amps in val_loader:
                inputs   = inputs.to(device)
                in_amps  = in_amps.to(device)
                targets  = targets.to(device)
                tgt_amps = tgt_amps.to(device)
                pw, pa = model(inputs, in_amps)
                loss, w_loss, a_loss, s_loss = criterion(pw, pa, targets, tgt_amps)
                val_loss['total'] += loss.item()
                val_loss['wave']  += w_loss.item()
                val_loss['amp']   += a_loss.item()
                val_loss['spec']  += s_loss.item()

        n_val = max(len(val_loader), 1)
        for k in val_loss:
            val_loss[k] /= n_val

        history['val_total'].append(val_loss['total'])
        history['val_wave'].append(val_loss['wave'])
        history['val_amp'].append(val_loss['amp'])
        history['val_spec'].append(val_loss['spec'])

        v_wave_c = w_wave * val_loss['wave']
        v_amp_c  = w_amp  * val_loss['amp']
        v_spec_c = w_spec * val_loss['spec']
        history['val_wave_contrib'].append(v_wave_c)
        history['val_amp_contrib'].append(v_amp_c)
        history['val_spec_contrib'].append(v_spec_c)

        # ---------- Print loss proportions ----------
        total_contrib = wave_c + amp_c + spec_c
        if total_contrib > 0:
            pct_w = wave_c / total_contrib * 100
            pct_a = amp_c / total_contrib * 100
            pct_s = spec_c / total_contrib * 100
        else:
            pct_w = pct_a = pct_s = 0.

        v_total_contrib = v_wave_c + v_amp_c + v_spec_c
        if v_total_contrib > 0:
            v_pct_w = v_wave_c / v_total_contrib * 100
            v_pct_a = v_amp_c / v_total_contrib * 100
            v_pct_s = v_spec_c / v_total_contrib * 100
        else:
            v_pct_w = v_pct_a = v_pct_s = 0.

        if verbose and ((epoch + 1) % 5 == 0 or epoch == 0):
            print(f"  Epoch {epoch+1}/{total_epochs}:")
            print(f"    Train total={epoch_loss['total']:.6f} | "
                  f"wave={epoch_loss['wave']:.4f}(×{w_wave}→{wave_c:.4f}, {pct_w:.1f}%) "
                  f"amp={epoch_loss['amp']:.4f}(×{w_amp}→{amp_c:.4f}, {pct_a:.1f}%) "
                  f"spec={epoch_loss['spec']:.4f}(×{w_spec}→{spec_c:.4f}, {pct_s:.1f}%)")
            print(f"    Val   total={val_loss['total']:.6f} | "
                  f"wave={val_loss['wave']:.4f}({v_pct_w:.1f}%) "
                  f"amp={val_loss['amp']:.4f}({v_pct_a:.1f}%) "
                  f"spec={val_loss['spec']:.4f}({v_pct_s:.1f}%)")

        # ---------- Early stopping ----------
        if val_loss['total'] < best_val_loss:
            best_val_loss = val_loss['total']
            no_improve_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            no_improve_count += 1
            if no_improve_count >= patience:
                if verbose:
                    print(f"  Early stopping at epoch {epoch+1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(save_dir, f'baseline_run_{run_id}.pth'))
    return model, best_val_loss, history


# ========================================================
# Plotting functions (with loss-proportion area plots)
# ========================================================
def plot_training_history(history, run_id, output_dir, w_wave=1.0, w_amp=0.1, w_spec=0.1):
    fig, axes = plt.subplots(3, 2, figsize=(16, 15))
    epochs_arr = list(range(1, len(history['train_total']) + 1))

    # (0,0) Total loss
    ax = axes[0, 0]
    ax.plot(epochs_arr, history['train_total'], label='Train Total', linewidth=1.5)
    ax.plot(epochs_arr, history['val_total'], label='Val Total', linewidth=1.5)
    val_arr = history['val_total']
    if val_arr:
        best_ep = int(np.argmin(val_arr)) + 1
        ax.axvline(x=best_ep, color='r', linestyle='--', alpha=0.4, label=f'Best@{best_ep}')
    ax.set_title('Total Loss (Train vs Val)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(True, alpha=0.3)

    # (0,1) Three loss components
    ax = axes[0, 1]
    for key, label, color in [('wave', 'Waveform', 'tab:blue'),
                                ('amp', 'Amplitude', 'tab:orange'),
                                ('spec', 'Spectral', 'tab:green')]:
        ax.plot(epochs_arr, history[f'train_{key}'], label=f'Train {label}',
                color=color, linewidth=1.5)
        ax.plot(epochs_arr, history[f'val_{key}'], label=f'Val {label}',
                color=color, linewidth=1.5, linestyle='--')
    ax.set_title('Individual Losses (solid=Train, dashed=Val)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (1,0) Training loss (stacked)
    ax = axes[1, 0]
    wc = np.array(history['train_wave_contrib'])
    ac = np.array(history['train_amp_contrib'])
    sc = np.array(history['train_spec_contrib'])
    ax.stackplot(epochs_arr, wc, ac, sc,
                 labels=[f'Wave (×{w_wave})', f'Amp (×{w_amp})', f'Spec (×{w_spec})'],
                 colors=['#4C72B0', '#DD8452', '#55A868'], alpha=0.8)
    ax.plot(epochs_arr, history['train_total'], 'k-', linewidth=1, label='Total')
    ax.set_title('Train: Weighted Contribution (Stacked)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(True, alpha=0.3)

    # (1,1) Validation loss (stacked)
    ax = axes[1, 1]
    wc = np.array(history['val_wave_contrib'])
    ac = np.array(history['val_amp_contrib'])
    sc = np.array(history['val_spec_contrib'])
    ax.stackplot(epochs_arr, wc, ac, sc,
                 labels=[f'Wave (×{w_wave})', f'Amp (×{w_amp})', f'Spec (×{w_spec})'],
                 colors=['#4C72B0', '#DD8452', '#55A868'], alpha=0.8)
    ax.plot(epochs_arr, history['val_total'], 'k-', linewidth=1, label='Total')
    ax.set_title('Val: Weighted Contribution (Stacked)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
    ax.legend(); ax.grid(True, alpha=0.3)

    # (2,0) Training loss proportion (%)
    ax = axes[2, 0]
    wc = np.array(history['train_wave_contrib'])
    ac = np.array(history['train_amp_contrib'])
    sc = np.array(history['train_spec_contrib'])
    tc = wc + ac + sc
    tc = np.where(tc == 0, 1, tc)
    ax.stackplot(epochs_arr, wc/tc*100, ac/tc*100, sc/tc*100,
                 labels=['Wave %', 'Amp %', 'Spec %'],
                 colors=['#4C72B0', '#DD8452', '#55A868'], alpha=0.8)
    ax.set_ylim(0, 100)
    ax.set_title('Train: Loss Proportion (%)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Percentage (%)')
    ax.legend(); ax.grid(True, alpha=0.3)

    # (2,1) Validation loss proportion (%)
    ax = axes[2, 1]
    wc = np.array(history['val_wave_contrib'])
    ac = np.array(history['val_amp_contrib'])
    sc = np.array(history['val_spec_contrib'])
    tc = wc + ac + sc
    tc = np.where(tc == 0, 1, tc)
    ax.stackplot(epochs_arr, wc/tc*100, ac/tc*100, sc/tc*100,
                 labels=['Wave %', 'Amp %', 'Spec %'],
                 colors=['#4C72B0', '#DD8452', '#55A868'], alpha=0.8)
    ax.set_ylim(0, 100)
    ax.set_title('Val: Loss Proportion (%)')
    ax.set_xlabel('Epoch'); ax.set_ylabel('Percentage (%)')
    ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle(f'Baseline Run {run_id} — Training History & Loss Decomposition',
                 fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'loss_run_{run_id}.png'), dpi=150, bbox_inches='tight')
    plt.close()


# ========================================================
# Part 7: Evaluation function (single-stage version)
# ========================================================
def evaluate_model(model, val_loader, device, w_wave=1.0, w_amp=0.1, w_spec=0.1):
    model.eval()
    criterion = SimpleDualOutputLoss(w_wave=w_wave, w_amp=w_amp, w_spec=w_spec)
    preds_w, preds_a, targs_w, targs_a = [], [], [], []
    total_loss = 0.0

    with torch.no_grad():
        for inputs, in_amps, targets, tgt_amps in val_loader:
            inputs   = inputs.to(device)
            in_amps  = in_amps.to(device)
            targets  = targets.to(device)
            tgt_amps = tgt_amps.to(device)
            pw, pa = model(inputs, in_amps)
            loss, _, _, _ = criterion(pw, pa, targets, tgt_amps)
            total_loss += loss.item()
            preds_w.append(pw.cpu()); preds_a.append(pa.cpu())
            targs_w.append(targets.cpu()); targs_a.append(tgt_amps.cpu())

    preds_w = torch.cat(preds_w); preds_a = torch.cat(preds_a)
    targs_w = torch.cat(targs_w); targs_a = torch.cat(targs_a)
    pred_orig = preds_w * (10 ** preds_a)
    targ_orig = targs_w * (10 ** targs_a)

    return {
        'val_loss':  total_loss / max(len(val_loader), 1),
        'wave_mse':  torch.mean((preds_w - targs_w) ** 2).item(),
        'wave_mae':  torch.mean(torch.abs(preds_w - targs_w)).item(),
        'amp_mse':   torch.mean((preds_a - targs_a) ** 2).item(),
        'amp_mae':   torch.mean(torch.abs(preds_a - targs_a)).item(),
        'orig_rmse': np.sqrt(torch.mean((pred_orig - targ_orig) ** 2).item()),
        'orig_mae':  torch.mean(torch.abs(pred_orig - targ_orig)).item(),
    }


# ========================================================
# Part 8: Main pipeline (single-stage version)
# ========================================================
def main_pipeline():

    CONFIG = {
        # ---- Data paths ----
        'train_data_dir': "autodl-tmp/train/111",
        'val_data_dir': "autodl-tmp/val/1",

        # ---- Station configuration (consistent with the fault-conditioned version for a fair comparison) ----
        'input_station_filter':  ['ALP', 'PLS', 'MOP', 'RIO'],
        'output_station_filter': None,


        # ---- Waveform components ----
        'input_component_dirs':  ['x', 'y'],
        'output_component_dirs': ['px', 'py'],

        # ---- Data parameters ----
        'time_steps': 25000,

        # ---- Model parameters (identical to the fault-conditioned version for a fair comparison) ----
        'hidden_channels': 16,
        'kernel_size':     (15, 1, 2),
        'window_size':     3000,
        'stride':          2000,
        'lstm_hidden':     24,
        'lstm_layers':     1,

        # ---- Single-stage training parameters ----
        'total_epochs':       80,       # total epochs (previously 4x10=40)
        'w_wave':             1.0,      # waveform weight
        'w_amp':              0.1,      # amplitude weight
        'w_spec':             0.1,      # spectral weight
        'lr':                 1e-3,
        'batch_size':         64,
        'accumulation_steps': 2,
        'patience':           80,
        'num_runs':           10,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"baseline_single_stage_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # ========== Step 1: Build the dataset ==========
    print("\n" + "=" * 70)
    print("Step 1: Loading waveform data")
    print("=" * 70)

    print("\n--- Training set ---")
    train_dataset, in_names, out_names = build_baseline_dataset(
        CONFIG['train_data_dir'],
        CONFIG['input_component_dirs'], CONFIG['output_component_dirs'],
        CONFIG['input_station_filter'], CONFIG['output_station_filter'],
        CONFIG['time_steps'], verbose=True, use_multiprocess=True)

    print("\n--- Validation set ---")
    val_dataset, _, _ = build_baseline_dataset(
        CONFIG['val_data_dir'],
        CONFIG['input_component_dirs'], CONFIG['output_component_dirs'],
        CONFIG['input_station_filter'], CONFIG['output_station_filter'],
        CONFIG['time_steps'], verbose=True, use_multiprocess=False)

    if train_dataset is None:
        print("Training set is empty!"); return
    if val_dataset is None:
        print("Validation set is empty; using the training set instead")
        val_dataset = train_dataset

    S_in  = train_dataset.inputs.shape[3]
    S_out = train_dataset.targets.shape[2]
    C     = train_dataset.inputs.shape[4]
    T     = train_dataset.inputs.shape[2]

    print(f"\n📊 S_in={S_in}, S_out={S_out}, C={C}, T={T}")
    print(f"   Training samples: {len(train_dataset)}, validation samples: {len(val_dataset)}")

    # ========== Step 2: Training ==========
    print("\n" + "=" * 70)
    print(f"Step 2: Training the baseline model (single-stage)")
    print(f"   Loss weights: w_wave={CONFIG['w_wave']}, "
          f"w_amp={CONFIG['w_amp']}, w_spec={CONFIG['w_spec']}")
    print("=" * 70)

    all_results = []
    for run_id in range(1, CONFIG['num_runs'] + 1):
        print(f"\n{'#'*70}\n# Run {run_id}/{CONFIG['num_runs']} (Random Initialization)\n{'#'*70}")

        train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                                  shuffle=True, num_workers=0)
        val_loader   = DataLoader(val_dataset,   batch_size=CONFIG['batch_size'],
                                  shuffle=False, num_workers=0)

        model = BaselineModel(
            input_channels=1,
            hidden_channels=CONFIG['hidden_channels'],
            kernel_size=CONFIG['kernel_size'],
            num_input_stations=S_in,
            num_output_stations=S_out,
            num_components=C,
            time_steps=T,
            window_size=CONFIG['window_size'],
            stride=CONFIG['stride'],
            lstm_hidden=CONFIG['lstm_hidden'],
            lstm_layers=CONFIG['lstm_layers'],
        )
        print(f"   Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Single-stage training
        model, best_val, history = train_model(
            model, train_loader, val_loader, CONFIG, device,
            run_id=run_id, save_dir=output_dir, verbose=True)

        # Evaluation
        metrics = evaluate_model(model, val_loader, device,
                                 w_wave=CONFIG['w_wave'],
                                 w_amp=CONFIG['w_amp'],
                                 w_spec=CONFIG['w_spec'])
        metrics['run_id']        = run_id
        metrics['seed']          = 'random'
        metrics['best_val_loss'] = best_val
        all_results.append(metrics)

        print(f"\n📊 Run {run_id}: val_loss={metrics['val_loss']:.6f}, "
              f"wave_mse={metrics['wave_mse']:.6f}, orig_rmse={metrics['orig_rmse']:.6f}")

        # Plot training history (including loss-proportion area plots)
        plot_training_history(history, run_id, output_dir,
                              w_wave=CONFIG['w_wave'],
                              w_amp=CONFIG['w_amp'],
                              w_spec=CONFIG['w_spec'])

        # Save history
        history_saveable = {k: [float(v) for v in vals] for k, vals in history.items()}
        with open(os.path.join(output_dir, f'history_run_{run_id}.json'), 'w') as f:
            json.dump(history_saveable, f, indent=2)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========== Step 3: Summary ==========
    print(f"\n{'='*70}\nSummary\n{'='*70}")

    vl  = [r['val_loss']  for r in all_results]
    wm  = [r['wave_mse']  for r in all_results]
    or_ = [r['orig_rmse'] for r in all_results]

    summary = {
        'model': 'BaselineModel (without fault embedding) - single-stage version',
        'seed_mode': 'random',
        'input_stations':  in_names,
        'output_stations': out_names,
        'loss_weights': {
            'w_wave': CONFIG['w_wave'],
            'w_amp':  CONFIG['w_amp'],
            'w_spec': CONFIG['w_spec'],
        },
        'results':    all_results,
        'statistics': {
            'val_loss_mean':  float(np.mean(vl)),  'val_loss_std':  float(np.std(vl)),
            'wave_mse_mean':  float(np.mean(wm)),  'wave_mse_std':  float(np.std(wm)),
            'orig_rmse_mean': float(np.mean(or_)), 'orig_rmse_std': float(np.std(or_)),
        },
    }
    with open(os.path.join(output_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    with open(os.path.join(output_dir, 'summary_report.txt'), 'w', encoding='utf-8') as f:
        f.write("Baseline model (without fault embedding) - single-stage training report\n")
        f.write(f"{'='*60}\n")
        f.write(f"Loss weights: w_wave={CONFIG['w_wave']}, "
                f"w_amp={CONFIG['w_amp']}, w_spec={CONFIG['w_spec']}\n")
        f.write(f"Input stations: {in_names}\nOutput stations: {out_names}\n")
        f.write(f"Training samples: {len(train_dataset)}, validation samples: {len(val_dataset)}\n\n")
        f.write(f"{'Run':<5} {'Val Loss':<12} {'Wave MSE':<12} {'Wave MAE':<12} "
                f"{'Amp MSE':<12} {'Amp MAE':<12} {'Orig RMSE':<12} {'Orig MAE':<12}\n")
        f.write(f"{'-'*89}\n")
        for r in all_results:
            f.write(f"{r['run_id']:<5} {r['val_loss']:<12.6f} {r['wave_mse']:<12.6f} "
                    f"{r['wave_mae']:<12.6f} {r['amp_mse']:<12.6f} {r['amp_mae']:<12.6f} "
                    f"{r['orig_rmse']:<12.6f} {r['orig_mae']:<12.6f}\n")
        f.write(f"\n{'='*60}\n")
        f.write(f"Mean Val Loss:   {np.mean(vl):.6f} +/- {np.std(vl):.6f}\n")
        f.write(f"Mean Wave MSE:   {np.mean(wm):.6f} +/- {np.std(wm):.6f}\n")
        f.write(f"Mean Orig RMSE:  {np.mean(or_):.6f} +/- {np.std(or_):.6f}\n")

    for r in all_results:
        print(f"  Run {r['run_id']}: val={r['val_loss']:.6f}, "
              f"mse={r['wave_mse']:.6f}, orig_rmse={r['orig_rmse']:.6f}")
    print(f"\n  Mean val_loss:  {np.mean(vl):.6f} ± {np.std(vl):.6f}")
    print(f"  Mean orig_rmse: {np.mean(or_):.6f} ± {np.std(or_):.6f}")
    print(f"Done! Results saved to: {output_dir}")


if __name__ == '__main__':
    main_pipeline()