Volume dataset saved to: graphleak_volume_experiments.csv
Scenario summary saved to: graphleak_volume_summary.csv

Dataset generated successfully.
Shape: (8700, 27)

Rows per scenario:
145    60
Name: count, dtype: int64

Scenario types:
scenario_type
normal             10
onset_leak         25
persistent_leak    25
Name: count, dtype: int64

Final label distribution by scenario type:
scenario_type    last_label
normal           0             10
onset_leak       1              5
                 2              5
                 3              5
                 4              5
                 5              5
persistent_leak  1              5
                 2              5
                 3              5
                 4              5
                 5              5
dtype: int64

Sample-level label distribution:
label
0    3250
1    1090
2    1090
3    1090
4    1090
5    1090
Name: count, dtype: int64

Sample-level binary leak distribution:
leak_binary
0    3250
1    5450
Name: count, dtype: int64

First rows:
   scenario_id source_file scenario_type  time_s  hour  V_N2  V_N3  ...  leak_downstream_N2  leak_downstream_N8  leak_downstream_N9  leak_x  leak_y  leak_z  weekday_flag
0            1  data_1.csv        normal       0     0  18.5   3.0  ...                   0                   0                   0     0.0     0.0     0.0             1
1            1  data_1.csv        normal     600     0  18.5   3.0  ...                   0                   0                   0     0.0     0.0     0.0             1
2            1  data_1.csv        normal    1200     0  18.5   3.0  ...                   0                   0                   0     0.0     0.0     0.0             1
3            1  data_1.csv        normal    1800     0  18.5   3.0  ...                   0                   0                   0     0.0     0.0     0.0             1
4            1  data_1.csv        normal    2400     0  18.5   3.0  ...                   0                   0                   0     0.0     0.0     0.0             1

[5 rows x 27 columns]
