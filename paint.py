import copy
import os
import pandas as pd
from scipy.interpolate import interpolate
from Const import TrainConsts
from Const.Consts import DEVICE_INFO
from datetime import datetime, timedelta
import matplotlib.pyplot as plt


def parse_sounding_file(config, obs_time):
    unified_format_file_name = \
        config["station_id"] + "_" + obs_time + ".txt"
    file_path = os.path.join(
        config["sounding_path"],
        config["station_id"], obs_time[:4], obs_time[4:6]
    )
    fullPath = os.path.join(file_path,
                            unified_format_file_name)
    print(f"EC：文件{fullPath}")
    # 开始观测时间
    # obsTime = ""
    # if re.match(SOUNDING_UNIFIED_RE_STRING,
    #             sounding_file.unified_format_file_name):
    #     obsTime = sounding_file.unified_format_file_name.split("_")[1][0:14]
    # 文件读取
    try:
        df = pd.read_csv(fullPath, sep=" ", skiprows=0, header=None, engine='python')
        for col in [0, 1, 2, 3]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        max_height = df[3].max()
        # 对缺测列进行插值
        df.interpolate(inplace=True)

        # 生成线性插值函数
        func_temperature = interpolate.interp1d(df[3],
                                                df[0],
                                                fill_value="extrapolate")
        func_pressure = interpolate.interp1d(df[3],
                                             df[1],
                                             fill_value="extrapolate")
        func_humidity = interpolate.interp1d(df[3],
                                             df[2],
                                             fill_value="extrapolate")

        meteorological_elements_83layers = []
        # 高度层对应的要素值模板
        meteorological_elements_83layer = {
            "height": None,
            "temperature": None,
            "pressure": None,
            "humidity": None,
        }
        # 设备海拔高度,探空数据插值时加上海拔高度
        alt = float(DEVICE_INFO[
                        config["station_id"]]["alt"])
        height83 = [(i * 1000 + round(alt)) for i in TrainConsts.BASE_HEIGHT83]
        last_pressure = -1
        last_temperature = -1
        for height in height83:
            if height > max_height:
                pass

            pressure = float(func_pressure(height))
            if pressure == last_pressure:
                pressure -= 0.001
            last_pressure = pressure

            temperature = float(func_temperature(height))
            if temperature == last_temperature:
                temperature -= 0.001
            last_temperature = temperature

            humidity = float(func_humidity(height))
            # 湿度插值异常值处理
            humidity = 0 if humidity < 0 else humidity
            humidity = 100 if humidity > 100 else humidity

            tmp_meteorological_elements_83layer = copy.deepcopy(
                meteorological_elements_83layer)
            tmp_meteorological_elements_83layer["height"] = height
            tmp_meteorological_elements_83layer["temperature"] = round(
                temperature, 3)
            tmp_meteorological_elements_83layer["humidity"] = round(
                humidity, 3)
            tmp_meteorological_elements_83layer["pressure"] = round(
                pressure, 3)
            meteorological_elements_83layers.append(
                tmp_meteorological_elements_83layer)
        return meteorological_elements_83layers
    except FileNotFoundError:
        print(f"缺少{obs_time}探空文件")


def parse_lv2_file(config, obs_time, kind):
    if kind == 'origin':
        filename = f"Z_UPAR_I_{config['station_id']}_{obs_time[:8]}000000_P_YMWR_6000A_CP_D.txt"
        full_path = os.path.join(config['ori_lv2_path'], obs_time[:8], filename)
        df = pd.read_csv(full_path, skiprows=2, encoding='gbk')
        df['DateTime'] = df['DateTime'].apply(lambda x: x[:-2] + '00')

        obs_time = datetime.strptime(obs_time, '%Y%m%d%H%M%S')
        obs_time = datetime.strftime(obs_time, '%Y-%m-%d %H:%M:%S')
        df = df[df['DateTime'] == obs_time]
        if df.empty:
            raise FileNotFoundError
        df = df.drop(['Record', 'SurTem(℃)', 'SurHum(%)', 'SurPre(hPa)', 'Tir(℃)', 'Rain', 'CloudBase(km)', 'Vint(mm)',
                      'Lqint(mm)', 'QCflag'], axis=1)
        print(f"原始LV2文件：{full_path}")

    elif kind == 'inversion':
        filename = f"Z_UPAR_I_{config['station_id']}_{obs_time[:8]}000000_P_YMWR_6000A_CP_D.txt"
        full_path = os.path.join(config['inversion_lv2_path'], filename)
        df = pd.read_csv(full_path, encoding='utf-8')

        df['DateTime'] = df['DateTime'].apply(lambda x: x + ':00')

        obs_time = datetime.strptime(obs_time, '%Y%m%d%H%M%S')
        obs_time = datetime.strftime(obs_time, '%Y/%m/%d %H:%M:%S')
        df = df[df['DateTime'] == obs_time]
        df = df.drop(['Record', 'SurTem(℃)', 'SurHum(%)', 'SurPre(hPa)', 'Tir(℃)', 'Rain',], axis=1)
        print(f"反演LV2文件：{full_path}")
    else:
        print('参数错误')
        return


    heights = []
    for height in df.columns[2:]:
        heights.append(float(height[:-4]) * 1000)
    return df, heights


def paint(station_id, obs_time):
    # 提取探空格式的ec数据
    world_time = datetime.strptime(obs_time, '%Y%m%d%H%M%S') + timedelta(hours=8)
    world_time = datetime.strftime(world_time, '%Y%m%d%H%M%S')
    parse_config = {
        'station_id': str(station_id),
        'sounding_path': r"D:/Data/microwave radiometer/Sounding",
        'ori_lv2_path': r"D:\Data\microwave radiometer\Measured brightness temperature\53996邯郸成安县",
        'inversion_lv2_path': r"D:\Data\microwave radiometer\LV2\53996"
    }
    # ec文件名为世界时间 = 北京时间+8h
    ec = parse_sounding_file(parse_config, obs_time=world_time)
    ec = pd.DataFrame(ec)
    if ec.empty:
        print("该时刻EC不存在")
        return

    # 提取原始Lv2数据
    try:
        origin_lv2, heights1 = parse_lv2_file(parse_config, obs_time, kind='origin')
    except FileNotFoundError:
        print("该时刻原始LV2文件或该时刻数据不存在")
        return

    # 提取反演得到的lv2数据
    try:
        inversion_lv2, heights2 = parse_lv2_file(parse_config, obs_time, kind='inversion')
    except FileNotFoundError:
        print("该时刻反演LV2文件不存在")
        return

    if not os.path.exists(rf'./out/{station_id}/{obs_time}'):
        os.makedirs(rf'./out/{station_id}/{obs_time}')

    # 绘温度图
    plt.figure(figsize=(7, 10))
    plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.plot(ec['temperature'].tolist(), ec['height'].tolist(), color='g', label='EC')
    plt.plot(origin_lv2[origin_lv2['10'] == 11].iloc[0, :].tolist()[2:], heights1, color='b', label='原始LV2')
    plt.plot(inversion_lv2[inversion_lv2['10'] == 11].iloc[0, :].tolist()[2:], heights2, label='反演获得的LV2')
    plt.legend(loc=0)
    plt.xlabel('温度，单位：℃')
    plt.ylabel('高度，单位：m')
    plt.text(x=0, y=10000, s=obs_time)
    plt.savefig(rf"./out/{station_id}/{obs_time}/temperature.jpg")
    plt.close()

    # 绘湿度图
    plt.figure(figsize=(7, 10))
    plt.rcParams['font.sans-serif'] = ['SimHei']    # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.plot(ec['humidity'].tolist(), ec['height'].tolist(), color='g', label='EC')
    plt.plot(origin_lv2[origin_lv2['10'] == 13].iloc[0, :].tolist()[2:], heights1, color='b', label='原始LV2')
    plt.plot(inversion_lv2[inversion_lv2['10'] == 13].iloc[0, :].tolist()[2:], heights2, label='反演获得的LV2')
    plt.legend(loc=0)
    plt.xlim(0, 100)
    plt.xlabel('湿度，单位：%')
    plt.ylabel('高度，单位：m')
    plt.text(x=0, y=10000, s=obs_time)
    plt.savefig(rf"./out/{station_id}/{obs_time}/humidity.jpg")
    plt.close()


if __name__ == '__main__':
    for root, _, files in os.walk(r"D:\Data\microwave radiometer\LV2"):
        for file in files:
            station = file.split('_')[3]
            obs_time = file.split('_')[4][:8] + '080000'
            paint(station, obs_time)
            obs_time = obs_time[:8] + '200000'
            paint(station, obs_time)
