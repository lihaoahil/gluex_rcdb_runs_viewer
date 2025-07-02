#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from sqlalchemy import create_engine
import rcdb
from rcdb.provider import RCDBProvider

# set up seaborn-like palette
palette = [
    "#1f77b4",  # dark blue for 0/90 PARA
    "#2ca02c",  # dark green for 45/135 PARA
    "#aec7e8",  # light blue for 0/90 PERP
    "#98df8a",  # light green for 45/135 PERP
    "#f08080"   # light coral red for AMO
]
polarization_map = {
    0.0: palette[0],
    45.0: palette[1],
    90.0: palette[2],
    135.0: palette[3],
    -1.0: palette[4]
}
pol_labels = {
    0.0: "0/90 PARA",
    45.0: "45/135 PARA",
    90.0: "0/90 PERP",
    135.0: "45/135 PERP",
    -1.0: "AMO"
}

def main():
    parser = argparse.ArgumentParser(description="Run summary and plots")
    parser.add_argument("-begin", type=int, required=True, help="begin run number")
    parser.add_argument("-end", type=int, required=True, help="end run number")
    args = parser.parse_args()
    
    start_run = args.begin
    end_run = args.end
    
    # connect to RCDB
    db = RCDBProvider("mysql://rcdb@hallddb.jlab.org/rcdb2")
    table = db.select_values(
        ['event_count', 'polarization_angle', 'beam_current', 'is_valid_run_end'], 
        run_min=start_run,
        run_max=end_run
    )
    
    # connect via sqlalchemy
    connection_url = "mysql+pymysql://rcdb@hallddb.jlab.org/rcdb2"
    engine = create_engine(connection_url)
    
    query = f"SELECT * FROM runs WHERE number >= {start_run} AND number <= {end_run}"
    df = pd.read_sql(query, engine)
    
    # convert
    df['started'] = pd.to_datetime(df['started'])
    df['finished'] = pd.to_datetime(df['finished'])
    df = df.dropna(subset=['started', 'finished'])
    
    df_cond = pd.DataFrame(table, columns=['number', 'event_count', 'polarization_angle', 'beam_current', 'is_valid_run_end'])
    df_merged = pd.merge(df, df_cond, on='number', how='left')
    df_merged = df_merged.sort_values('started')
    
    df_merged['cum_events'] = df_merged['event_count'].cumsum()
    df_merged['bottom'] = df_merged['cum_events'] - df_merged['event_count']
    
    summary = df_merged.groupby('polarization_angle').agg(
        total_events=('event_count', 'sum'),
        runs=('number', lambda x: list(sorted(x.unique())))
    )
    summary['total_events_million'] = summary['total_events'] / 1e6
    summary['total_events_billion'] = summary['total_events'] / 1e9
    
    overall_time_min = df_merged['started'].min()
    overall_time_max = df_merged['started'].max()
    
    print(f"Time range: {overall_time_min.strftime('%Y-%m-%d %H:%M:%S')} to {overall_time_max.strftime('%Y-%m-%d %H:%M:%S')}")
    
    total_trigger = 0
    for pol, row in summary.iterrows():
        desc = pol_labels.get(pol, str(pol))
        triggers_billion = round(row['total_events'] / 1e9, 2)
        total_trigger += triggers_billion
        run_list = sorted(int(r) for r in row['runs'])
        run_str = ", ".join(str(r) for r in run_list)
        print(f"* {desc}: {triggers_billion:.2f}B triggers")
        print(f"  Runs: [{run_str}]")
    print(f"\nTotal triggers: {total_trigger:.2f}B\n")
    
    # -------- Figure 1: horizontal lines ---------
    plt.figure(figsize=(10, 6))
    plt.hlines(
        y=df['number'],
        xmin=df['started'],
        xmax=df['finished'],
        color="blue",
        linewidth=6
    )
    plt.xlabel("Time")
    plt.ylabel("Run Number")
    plt.title("Run durations (no overlap)")
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x):d}"))
    plt.tight_layout()
    plt.savefig(f"run_durations_{start_run}_{end_run}.png", dpi=300)
    plt.close()
    
    # -------- Figure 2: bar accumulation ---------
    fig, ax = plt.subplots(figsize=(18, 10))
    for _, row in df_merged.iterrows():
        color = polarization_map.get(row['polarization_angle'], 'gray')
        ax.bar(
            x=row['started'],
            height=row['event_count'],
            width=(row['finished'] - row['started']).total_seconds()/(24*3600),
            bottom=row['bottom'],
            color=color,
            edgecolor='black' if row['is_valid_run_end'] else 'red',
            linewidth=4.0,
            align='edge'
        )
    
    unique_days = pd.to_datetime(df_merged['started']).dt.floor('D').unique()
    ymax = df_merged['cum_events'].max() * 1.05
    tick_positions, tick_labels = [], []
    for day in unique_days:
        for shift_hour in [0, 8, 16]:
            tick_time = day + pd.Timedelta(hours=shift_hour)
            tick_num = mdates.date2num(tick_time)
            ax.axvline(x=tick_num, color='black', linestyle='--', alpha=0.5)
            tick_positions.append(tick_num)
            tick_labels.append(tick_time.strftime("%a %m-%d\n%H:%M"))
        ax.annotate('owl', xy=(mdates.date2num(day + pd.Timedelta(hours=4)), ymax*0.95),
                    ha='center', fontsize=18, color='black')
        ax.annotate('day', xy=(mdates.date2num(day + pd.Timedelta(hours=12)), ymax*0.95),
                    ha='center', fontsize=18, color='black')
        ax.annotate('swing', xy=(mdates.date2num(day + pd.Timedelta(hours=20)), ymax*0.95),
                    ha='center', fontsize=18, color='black')
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=11)
    
    legend_elements = []
    for pol, row in summary.iterrows():
        color = polarization_map.get(pol, 'gray')
        desc = pol_labels.get(pol, str(pol))
        triggers_billion = round(row['total_events'] / 1e9, 2)
        label = f"{desc:<15} {triggers_billion:>6.2f}B triggers"
        legend_elements.append(Line2D([0], [0], color=color, lw=6, label=label))
    
    ax.legend(
        handles=legend_elements,
        loc='lower right',
        fontsize=20,
        title="Polarization Angle",
        title_fontsize=20,
        frameon=True
    ).get_frame().set_facecolor('white')
    
    ax.set_ylabel("Accumulated Trigger Count", fontsize=16)
    ax.set_xlabel("Time", fontsize=16)
    ax.set_title("Weekend Production Summary", fontsize=20)
    ax.set_xlim(left=df_merged['started'].min(), right=df_merged['finished'].max())
    plt.tight_layout()
    plt.savefig(f"production_summary_{start_run}_{end_run}.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    main()
