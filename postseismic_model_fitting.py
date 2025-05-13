import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os
from matplotlib.gridspec import GridSpec

def exp_model(t, A, B, C, tau):
    return A + B * t + C * np.exp(-t / tau)

def log_model(t, A, B, C, tau):
    return A + B * t + C * np.log10(1 + t / tau)

def load_data(file_path):
    df = pd.read_csv(file_path, header=None)
    df.columns = ['Day', 'd-Lat', 'd-Lon', 'd-Hei']
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    
    return df

def objective_function(params, t, lat, lon, hei, model_func):
    A_lat, B_lat, C_lat, A_lon, B_lon, C_lon, A_hei, B_hei, C_hei, tau = params
    
    lat_pred = model_func(t, A_lat, B_lat, C_lat, tau)
    lon_pred = model_func(t, A_lon, B_lon, C_lon, tau)
    hei_pred = model_func(t, A_hei, B_hei, C_hei, tau)
    
    lat_residuals = lat - lat_pred
    lon_residuals = lon - lon_pred
    hei_residuals = hei - hei_pred
    
    return np.sum(np.square(lat_residuals)) + np.sum(np.square(lon_residuals)) + np.sum(np.square(hei_residuals))

def fit_models(df, station_name):
    t = np.array(df['Day'].values, dtype=float)
    lat = np.array(df['d-Lat'].values, dtype=float)
    lon = np.array(df['d-Lon'].values, dtype=float)
    hei = np.array(df['d-Hei'].values, dtype=float)
    
    initial_params_exp_model = [0.0, 0.001, 0.1, 0.0, 0.001, 0.1, 0.0, 0.001, 0.1, 100.0]
    initial_params_log_model = [0.0, 0.001, 0.1, 0.0, 0.001, 0.1, 0.0, 0.001, 0.1, 100.0]
    
    result_exp_model = minimize(
        lambda p: objective_function(p, t, lat, lon, hei, exp_model),
        initial_params_exp_model,
        method='CG',
        options={'maxiter': 10000}
    )
    params_exp_model = result_exp_model.x
    
    result_log_model = minimize(
        lambda p: objective_function(p, t, lat, lon, hei, log_model),
        initial_params_log_model,
        method='CG',
        options={'maxiter': 10000}
    )
    params_log_model = result_log_model.x

    A_lat1, B_lat1, C_lat1, A_lon1, B_lon1, C_lon1, A_hei1, B_hei1, C_hei1, tau1 = params_exp_model
    A_lat2, B_lat2, C_lat2, A_lon2, B_lon2, C_lon2, A_hei2, B_hei2, C_hei2, tau2 = params_log_model
    
    return {
        'station': station_name,
        'exp_model': {
            'params': {
                'lat': {'A': A_lat1, 'B': B_lat1, 'C': C_lat1},
                'lon': {'A': A_lon1, 'B': B_lon1, 'C': C_lon1},
                'hei': {'A': A_hei1, 'B': B_hei1, 'C': C_hei1},
                'tau': tau1
            },
            'fun': result_exp_model.fun
        },
        'log_model': {
            'params': {
                'lat': {'A': A_lat2, 'B': B_lat2, 'C': C_lat2},
                'lon': {'A': A_lon2, 'B': B_lon2, 'C': C_lon2},
                'hei': {'A': A_hei2, 'B': B_hei2, 'C': C_hei2},
                'tau': tau2
            },
            'fun': result_log_model.fun
        }
    }

def model_enhanced(t, A, B, C_exp, tau_exp, C_log, tau_log):
    return A + B * t + C_exp * np.exp(-t / tau_exp) + C_log * np.log10(1 + t / tau_log)

def objective_model_enhanced(params, t, lat, lon, hei):
    A_lat, B_lat, C_exp_lat, A_lon, B_lon, C_exp_lon, A_hei, B_hei, C_exp_hei, \
    C_log_lat, C_log_lon, C_log_hei, tau_exp, tau_log = params
    
    lat_pred = model_enhanced(t, A_lat, B_lat, C_exp_lat, tau_exp, C_log_lat, tau_log)
    lon_pred = model_enhanced(t, A_lon, B_lon, C_exp_lon, tau_exp, C_log_lon, tau_log)
    hei_pred = model_enhanced(t, A_hei, B_hei, C_exp_hei, tau_exp, C_log_hei, tau_log)
    
    lat_residuals = lat - lat_pred
    lon_residuals = lon - lon_pred
    hei_residuals = hei - hei_pred
    
    return np.sum(np.square(lat_residuals)) + np.sum(np.square(lon_residuals)) + np.sum(np.square(hei_residuals))

def fit_enhanced_model(df, station_name, basic_results):
    t = np.array(df['Day'].values, dtype=float)
    lat = np.array(df['d-Lat'].values, dtype=float)
    lon = np.array(df['d-Lon'].values, dtype=float)
    hei = np.array(df['d-Hei'].values, dtype=float)

    A_lat = 0.0
    B_lat = 0.001
    C_exp_lat = 0.1
    A_lon = 0.0
    B_lon = 0.001
    C_exp_lon = 0.1
    A_hei = 0.0
    B_hei = 0.001
    C_exp_hei = 0.1
    C_log_lat = 0.1
    C_log_lon = 0.1
    C_log_hei = 0.1
    tau_exp = 100.0
    tau_log = 100.0
    
    initial_params = [A_lat, B_lat, C_exp_lat, A_lon, B_lon, C_exp_lon, A_hei, B_hei, C_exp_hei,
                      C_log_lat, C_log_lon, C_log_hei, tau_exp, tau_log]
    
    result_model_enhanced = minimize(
        lambda p: objective_model_enhanced(p, t, lat, lon, hei),
        initial_params,
        method='CG',
        options={'maxiter': 10000}
    )
    params_enhanced = result_model_enhanced.x
    
    A_lat, B_lat, C_exp_lat, A_lon, B_lon, C_exp_lon, A_hei, B_hei, C_exp_hei, \
    C_log_lat, C_log_lon, C_log_hei, tau_exp, tau_log = params_enhanced
    
    return {
        'station': station_name,
        'enhanced_model': {
            'params': {
                'lat': {'A': A_lat, 'B': B_lat, 'C_exp': C_exp_lat, 'C_log': C_log_lat},
                'lon': {'A': A_lon, 'B': B_lon, 'C_exp': C_exp_lon, 'C_log': C_log_lon},
                'hei': {'A': A_hei, 'B': B_hei, 'C_exp': C_exp_hei, 'C_log': C_log_hei},
                'tau_exp': tau_exp,
                'tau_log': tau_log
            },
            'fun': result_model_enhanced.fun
        }
    }

def plot_three_models(df, station_name, basic_results, enhanced_results):
    t = np.array(df['Day'].values, dtype=float)
    lat = np.array(df['d-Lat'].values, dtype=float)
    lon = np.array(df['d-Lon'].values, dtype=float)
    hei = np.array(df['d-Hei'].values, dtype=float)
    
    exp_model_params = basic_results['exp_model']['params']
    log_model_params = basic_results['log_model']['params']
    enhanced_params = enhanced_results['enhanced_model']['params']
    t_smooth = np.linspace(min(t), max(t), 1000)
    
    lat_pred_exp = exp_model(
        t_smooth,
        exp_model_params['lat']['A'],
        exp_model_params['lat']['B'],
        exp_model_params['lat']['C'],
        exp_model_params['tau']
    )
    lon_pred_exp = exp_model(
        t_smooth,
        exp_model_params['lon']['A'],
        exp_model_params['lon']['B'],
        exp_model_params['lon']['C'],
        exp_model_params['tau']
    )
    hei_pred_exp = exp_model(
        t_smooth,
        exp_model_params['hei']['A'],
        exp_model_params['hei']['B'],
        exp_model_params['hei']['C'],
        exp_model_params['tau']
    )
    
    lat_pred_log = log_model(
        t_smooth,
        log_model_params['lat']['A'],
        log_model_params['lat']['B'],
        log_model_params['lat']['C'],
        log_model_params['tau']
    )
    lon_pred_log = log_model(
        t_smooth,
        log_model_params['lon']['A'],
        log_model_params['lon']['B'],
        log_model_params['lon']['C'],
        log_model_params['tau']
    )
    hei_pred_log = log_model(
        t_smooth,
        log_model_params['hei']['A'],
        log_model_params['hei']['B'],
        log_model_params['hei']['C'],
        log_model_params['tau']
    )
    
    lat_pred_enh = model_enhanced(
        t_smooth,
        enhanced_params['lat']['A'],
        enhanced_params['lat']['B'],
        enhanced_params['lat']['C_exp'],
        enhanced_params['tau_exp'],
        enhanced_params['lat']['C_log'],
        enhanced_params['tau_log']
    )
    lon_pred_enh = model_enhanced(
        t_smooth,
        enhanced_params['lon']['A'],
        enhanced_params['lon']['B'],
        enhanced_params['lon']['C_exp'],
        enhanced_params['tau_exp'],
        enhanced_params['lon']['C_log'],
        enhanced_params['tau_log']
    )
    hei_pred_enh = model_enhanced(
        t_smooth,
        enhanced_params['hei']['A'],
        enhanced_params['hei']['B'],
        enhanced_params['hei']['C_exp'],
        enhanced_params['tau_exp'],
        enhanced_params['hei']['C_log'],
        enhanced_params['tau_log']
    )
    
    lat_data_color = 'darkblue'
    lat_line_color = 'skyblue'
    lon_data_color = 'darkred'
    lon_line_color = 'lightcoral'
    hei_data_color = 'darkgreen'
    hei_line_color = 'lightgreen'
    
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(3, 3, figure=fig)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.scatter(t, lat, color=lat_data_color, s=30, alpha=0.7, label='Data', zorder=1)
    ax1.plot(t_smooth, lat_pred_exp, color=lat_line_color, linewidth=2.5, label='Model', zorder=2)
    ax1.set_title(f'Latitude - Exp-Model')
    ax1.set_xlabel('Days after earthquake (0=2011/3/12)')
    ax1.set_ylabel('Latitude displacement')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.scatter(t, lon, color=lon_data_color, s=30, alpha=0.7, label='Data', zorder=1)
    ax2.plot(t_smooth, lon_pred_exp, color=lon_line_color, linewidth=2.5, label='Model', zorder=2)
    ax2.set_title(f'Longitude - Exp-Model')
    ax2.set_xlabel('Days after earthquake (0=2011/3/12)')
    ax2.set_ylabel('Longitude displacement')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.scatter(t, hei, color=hei_data_color, s=30, alpha=0.7, label='Data', zorder=1)
    ax3.plot(t_smooth, hei_pred_exp, color=hei_line_color, linewidth=2.5, label='Model', zorder=2)
    ax3.set_title(f'Height - Exp-Model')
    ax3.set_xlabel('Days after earthquake (0=2011/3/12)')
    ax3.set_ylabel('Height displacement')
    ax3.legend()
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    ax4 = fig.add_subplot(gs[0, 1])
    ax4.scatter(t, lat, color=lat_data_color, s=30, alpha=0.7, label='Data', zorder=1)
    ax4.plot(t_smooth, lat_pred_log, color=lat_line_color, linewidth=2.5, label='Model', zorder=2)
    ax4.set_title(f'Latitude - Log-Model')
    ax4.set_xlabel('Days after earthquake (0=2011/3/12)')
    ax4.set_ylabel('Latitude displacement')
    ax4.legend()
    ax4.grid(True, linestyle='--', alpha=0.5)
    
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.scatter(t, lon, color=lon_data_color, s=30, alpha=0.7, label='Data', zorder=1)
    ax5.plot(t_smooth, lon_pred_log, color=lon_line_color, linewidth=2.5, label='Model', zorder=2)
    ax5.set_title(f'Longitude - Log-Model')
    ax5.set_xlabel('Days after earthquake (0=2011/3/12)')
    ax5.set_ylabel('Longitude displacement')
    ax5.legend()
    ax5.grid(True, linestyle='--', alpha=0.5)
    
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.scatter(t, hei, color=hei_data_color, s=30, alpha=0.7, label='Data', zorder=1)
    ax6.plot(t_smooth, hei_pred_log, color=hei_line_color, linewidth=2.5, label='Model', zorder=2)
    ax6.set_title(f'Height - Log-Model')
    ax6.set_xlabel('Days after earthquake (0=2011/3/12)')
    ax6.set_ylabel('Height displacement')
    ax6.legend()
    ax6.grid(True, linestyle='--', alpha=0.5)
    
    ax7 = fig.add_subplot(gs[0, 2])
    ax7.scatter(t, lat, color=lat_data_color, s=30, alpha=0.7, label='Data', zorder=1)
    ax7.plot(t_smooth, lat_pred_enh, color=lat_line_color, linewidth=2.5, label='Enhanced Model', zorder=2)
    ax7.set_title(f'Latitude - Enhanced Model')
    ax7.set_xlabel('Days after earthquake (0=2011/3/12)')
    ax7.set_ylabel('Latitude displacement')
    ax7.legend()
    ax7.grid(True, linestyle='--', alpha=0.5)
    
    ax8 = fig.add_subplot(gs[1, 2])
    ax8.scatter(t, lon, color=lon_data_color, s=30, alpha=0.7, label='Data', zorder=1)
    ax8.plot(t_smooth, lon_pred_enh, color=lon_line_color, linewidth=2.5, label='Enhanced Model', zorder=2)
    ax8.set_title(f'Longitude - Enhanced Model')
    ax8.set_xlabel('Days after earthquake (0=2011/3/12)')
    ax8.set_ylabel('Longitude displacement')
    ax8.legend()
    ax8.grid(True, linestyle='--', alpha=0.5)
    
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.scatter(t, hei, color=hei_data_color, s=30, alpha=0.7, label='Data', zorder=1)
    ax9.plot(t_smooth, hei_pred_enh, color=hei_line_color, linewidth=2.5, label='Enhanced Model', zorder=2)
    ax9.set_title(f'Height - Enhanced Model')
    ax9.set_xlabel('Days after earthquake (0=2011/3/12)')
    ax9.set_ylabel('Height displacement')
    ax9.legend()
    ax9.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f'{station_name}_models.png', dpi=300)
    plt.close()

def save_results_to_csv(all_results, enhanced_results):
    for result in all_results:
        station = result['station']
        exp_model_params = result['exp_model']['params']
        exp_model_data = {
            'Parameter': [
                'Station',
                'A_lat', 'B_lat', 'C_lat',
                'A_lon', 'B_lon', 'C_lon',
                'A_hei', 'B_hei', 'C_hei',
                'tau'
            ],
            'Value': [
                station,
                exp_model_params['lat']['A'], exp_model_params['lat']['B'], exp_model_params['lat']['C'],
                exp_model_params['lon']['A'], exp_model_params['lon']['B'], exp_model_params['lon']['C'],
                exp_model_params['hei']['A'], exp_model_params['hei']['B'], exp_model_params['hei']['C'],
                exp_model_params['tau']
            ]
        }
        pd.DataFrame(exp_model_data).to_csv('exp_model_results.csv', index=False)
    
    for result in all_results:
        station = result['station']
        log_model_params = result['log_model']['params']
        log_model_data = {
            'Parameter': [
                'Station',
                'A_lat', 'B_lat', 'C_lat',
                'A_lon', 'B_lon', 'C_lon',
                'A_hei', 'B_hei', 'C_hei',
                'tau'
            ],
            'Value': [
                station,
                log_model_params['lat']['A'], log_model_params['lat']['B'], log_model_params['lat']['C'],
                log_model_params['lon']['A'], log_model_params['lon']['B'], log_model_params['lon']['C'],
                log_model_params['hei']['A'], log_model_params['hei']['B'], log_model_params['hei']['C'],
                log_model_params['tau']
            ]
        }
        pd.DataFrame(log_model_data).to_csv('log_model_results.csv', index=False)
    
    for result in enhanced_results:
        station = result['station']
        params = result['enhanced_model']['params']
        enhanced_data = {
            'Parameter': [
                'Station',
                'A_lat', 'B_lat', 'C_exp_lat', 'C_log_lat',
                'A_lon', 'B_lon', 'C_exp_lon', 'C_log_lon',
                'A_hei', 'B_hei', 'C_exp_hei', 'C_log_hei',
                'tau_exp', 'tau_log'
            ],
            'Value': [
                station,
                params['lat']['A'], params['lat']['B'], params['lat']['C_exp'], params['lat']['C_log'],
                params['lon']['A'], params['lon']['B'], params['lon']['C_exp'], params['lon']['C_log'],
                params['hei']['A'], params['hei']['B'], params['hei']['C_exp'], params['hei']['C_log'],
                params['tau_exp'], params['tau_log']
            ]
        }
        pd.DataFrame(enhanced_data).to_csv('enhanced_model_results.csv', index=False)

all_results = []
enhanced_results = []
file = 'naruko_data.csv'
station_name = 'naruko'
df = load_data(file)
results = fit_models(df, station_name)
all_results.append(results)
enhanced_result = fit_enhanced_model(df, station_name, results)
enhanced_results.append(enhanced_result)
plot_three_models(df, station_name, results, enhanced_result)
save_results_to_csv(all_results, enhanced_results)