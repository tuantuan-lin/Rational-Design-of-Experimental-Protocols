# coding=utf-8
import importlib
import os
import sys

import cv2

importlib.reload(sys)
import csv
import matplotlib.pyplot as plt
import numpy
from scipy import optimize
import pandas as pd
import numpy as np
from matplotlib.patches import Ellipse
from matplotlib.text import OffsetFrom
from matplotlib import rcParams


def f_line(x, A, B):
    return A * x + B


class Data_process:
    def __init__(self):
        self.label = {'x': [], 'y': []}
        self.first_derivat = {'dx': [], 'dy': []}
        self.second_derivat = {'d2x': [], 'd2y': []}
        self.cout = 0
        self.keypoint_x = []
        self.keypoint_y = []
        self.keypoint_second_x = []
        self.keypoint_second_y = []
        self.sup = 350
        self.inf = 250

    def data_read(self, file_name, file_path):

        self.path = file_name.split('.')[0]
        with open(file_path) as f:
            fcsv = csv.reader(f)
            header_ = next(fcsv)
            for i in fcsv:
                if float(i[0]) not in self.label['x']:
                    self.label['x'].append(float(i[0]))
                    self.label['y'].append(float(i[1]))
                if float(i[0]) > 500:
                    break

    def data_derivat_first(self):
        if len(self.label['x']) <= 1:
            print('There is not enough data or not perform data_read')
            return None
        for i in range(len(self.label['x']) - 1):
            self.first_derivat['dx'].append((self.label['x'][i] + self.label['x'][i]) / 2)
            self.first_derivat['dy'].append(
                (self.label['y'][i + 1] - self.label['y'][i]) / (self.label['x'][i + 1] - self.label['x'][i]))

        return self.first_derivat

    def data_derivat_second(self):
        if len(self.first_derivat['dx']) <= 1:
            print('There are not perform data_derivat_first')
            return None
        for i in range(len(self.first_derivat['dx']) - 1):
            self.second_derivat['d2x'].append((self.first_derivat['dx'][i] + self.first_derivat['dx'][i]) / 2)
            self.second_derivat['d2y'].append((self.first_derivat['dy'][i + 1] - self.first_derivat['dy'][i]) / (
                    self.first_derivat['dx'][i + 1] - self.first_derivat['dx'][i] + 1e-5))
        return self.second_derivat

    def data_plot(self, res_path):
        self.x_first = numpy.array(self.first_derivat['dx'])
        self.y_first = numpy.array(self.first_derivat['dy'])
        first_max = self.y_first.max()
        first_min = self.y_first.min()

        self.x_second = numpy.array(self.second_derivat['d2x'])
        self.y_second = numpy.array(self.second_derivat['d2y'])

        second_max = self.y_second.max()
        second_min = self.y_second.min()
        middle_value_second = second_min / 3

        for i in range(1, len(self.x_second) - 1):
            if self.y_second[i] > -6:
                if self.y_second[i - 1] > self.y_second[i] and self.y_second[i] < self.y_second[i + 1] \
                        and self.y_second[i] < middle_value_second:
                    self.cout = self.cout + 1
                    if self.inf < self.x_second[i] < self.sup:
                        x = self.x_second[i]
                        min_ = 1000
                        x_index = 0
                        for i in range(len(self.label['x'])):
                            if abs(self.label['x'][i] - x) < min_:
                                min_ = abs(self.label['x'][i] - x)
                                x_index = i
                        self.keypoint_second_x.append(self.label['x'][x_index])
                        self.keypoint_second_y.append(self.label['y'][x_index])

        if self.cout == 0:
            print("The minimum of the second derivative is not found in {}".format(self.path))
            middle_value_first = (first_max + first_min) / 2
            for i in range(1, len(self.x_first) - 1):
                if self.y_first[i - 1] > self.y_first[i] and self.y_first[i] < self.y_first[i + 1] and self.y_first[
                    i] < middle_value_first:

                    if self.inf < self.x_first[i] < self.sup:
                        x = self.x_first[i]
                        min_ = 1000
                        x_index = 0
                        for i in range(len(self.label['x'])):
                            if abs(self.label['x'][i] - x) < min_:
                                min_ = abs(self.label['x'][i] - x)
                                x_index = i
                        self.keypoint_second_x.append(self.label['x'][x_index])
                        self.keypoint_second_y.append(self.label['y'][x_index])

        plt.clf()
        x_ori = numpy.array(self.label['x'])
        y_ori = numpy.array(self.label['y'])
        plt.plot(x_ori, y_ori)
        plt.plot(self.keypoint_second_x, self.keypoint_second_y, 'r+')
        plt.title('original data keypoint :{}'.format(self.cout))
        plt.savefig(res_path + '{}_0_ori_data.jpg'.format(self.path))
        plt.clf()

        x_first = self.x_first
        y_first = self.y_first

        x_second = numpy.array(self.second_derivat['d2x'])
        y_second = numpy.array(self.second_derivat['d2y'])

        figure, (ax1, ax2, ax3) = plt.subplots(3, 1,
                                               figsize=(5, 6),
                                               dpi=600,
                                               sharex=True)

        ax1.plot(x_ori, y_ori, c='blue', label='origial data')
        ax2.plot(x_first, y_first, c='orange', label='1st derivat')
        ax3.plot(x_second, y_second, c='r', label='2nd derivat')
        ax1.plot(self.keypoint_second_x, self.keypoint_second_y, 'r+')

        ax1.set_xlim(250, 380)
        ax2.set_ylim(first_min, first_max)
        ax3.set_ylim(second_min, second_max)

        ax1.set_ylabel("Weight")
        ax2.set_ylabel("1st derivat")
        ax3.set_ylabel("2nd derivat")

        figure.subplots_adjust(hspace=0.1)

        ax1.set_title('origial data')
        plt.savefig(res_path + '{}_3_process.jpg'.format(self.path))

        keypoint_sec_x = self.keypoint_second_x
        keypoint_sec_y = self.keypoint_second_y
        print('{}'.format(self.path))
        print('keypoint_sec_x:', keypoint_sec_x)
        print('keypoint_sec_y:', keypoint_sec_y)

    def data_tangent(self, res_path, df):

        tangent_1 = {'x': [], 'y': []}
        for i in range(len(self.label['x'])):
            if self.label['x'][i] > self.keypoint_second_x[0] - 50 and self.label['x'][i] < self.keypoint_second_x[0] \
                    and len(tangent_1['x']) < 3:
                tangent_1['x'].append(self.label['x'][i])
                tangent_1['y'].append(self.label['y'][i])

        print('Tangent point of the first tangent line：', tangent_1)

        print('self.keypoint_sec_x[0]', self.keypoint_second_x[0])
        temp_point_x = (tangent_1['x'][0] + self.keypoint_second_x[0]) / 2

        tangent_2 = {'x': [], 'y': []}

        min_ = 50

        res_point_x = 0
        res_point_y = 0
        for i in range(len(self.label['x'])):
            if abs(self.label['x'][i] - temp_point_x) < min_:
                min_ = abs(self.label['x'][i] - temp_point_x)
                res_point_x = self.label['x'][i]
                res_point_y = self.label['y'][i]
            if self.label['x'][i] > self.keypoint_second_x[0] and len(tangent_2['x']) < 4:
                tangent_2['x'].append(self.label['x'][i])
                tangent_2['y'].append(self.label['y'][i])

        if res_point_x != 0 and res_point_x not in tangent_1['x']:
            tangent_1['x'].append(res_point_x)
            tangent_1['y'].append(res_point_y)

        print('Unique tangent point:', tangent_1)


        if len(tangent_1['x']) < 2:
            tangent_1['x'].append(self.keypoint_second_x[0])
            tangent_1['y'].append(self.keypoint_second_y[0])

        A1, B1 = optimize.curve_fit(f_line, tangent_1['x'], tangent_1['y'])[0]
        A2, B2 = optimize.curve_fit(f_line, tangent_2['x'], tangent_2['y'])[0]

        plot_x = numpy.arange(275, 400, 1)
        plot_y1 = A1 * plot_x + B1
        plot_y2 = A2 * plot_x + B2
        min_point_y = 1000
        tangent_point = {'x': [0], 'y': [0]}
        for i in range(len(plot_x)):
            if abs(plot_y2[i] - plot_y1[i]) < min_point_y:
                min_point_y = abs(plot_y2[i] - plot_y1[i])
                tangent_point['x'][0] = plot_x[i]
                tangent_point['y'][0] = plot_y1[i]
        plt.clf()

        print('Thermal Decomposition Temperature:', tangent_point['x'], tangent_point['y'])
        plt.plot(self.label['x'], self.label['y'], c='blue')
        plt.plot(plot_x, plot_y1, alpha=0.8, c='orange')
        plt.plot(plot_x, plot_y2, alpha=0.8, c='r')

        plt.plot(tangent_point['x'], tangent_point['y'], '*', c='blue')
        plt.plot(tangent_1['x'], tangent_1['y'], color='orange', marker='o', fillstyle='none', markeredgecolor='brown')
        plt.plot(tangent_2['x'], tangent_2['y'], color='r', marker='o', fillstyle='none', markeredgecolor='r')

        plt.xlim(0, self.label['x'][-1] + 10)
        plt.ylim(self.label['y'][-1] - 10, self.label['y'][0] + 10)
        plt.title('Automatic extraction of \n thermal decomposition temperature', fontweight='bold')

        name = self.path
        tem = tangent_point['x'][0]
        y1 = tangent_point['y'][0]
        final_value = 400
        final_point_y = 0
        min_ = 50

        for i in range(len(self.label['x'])):
            if abs(self.label['x'][i] - final_value) < min_:
                min_ = abs(self.label['x'][i] - final_value)
                final_point_x = self.label['x'][i]
                final_point_y = self.label['y'][i]

        x0_point = np.array([0, self.label['x'][-1]])
        y0_point = np.array([final_point_y * 1.86, final_point_y * 1.86])
        x1_point = np.array([0, tem])
        y1_point = np.array([y1, y1])
        x2_point = np.array([0, final_point_x])
        y2_point = np.array([final_point_y, final_point_y])
        plt.plot(x0_point, y0_point, linestyle='dotted', color='black')
        plt.plot(x1_point, y1_point, linestyle='dotted', color='black')
        plt.plot(x2_point, y2_point, linestyle='dotted', color='black')

        defect = (1.86 - y1 / final_point_y) / 0.86
        plt.annotate('Ce6O6(BDC)6',
                     xy=(350, final_point_y * 1.86 + 1), xycoords='data')
        plt.annotate('Ce6O(6+x)(BDC)(6-x)',
                     xy=(50, y1 - 5), xycoords='data')
        plt.annotate('CeO2',
                     xy=(150, final_point_y + 1), xycoords='data')
        plt.annotate('Wloss={:.2%}'.format(defect),
                     xy=(80, y1 + 1), xycoords='data')

        plt.annotate('', xy=(50, y1), xycoords='data',
                     xytext=(50, final_point_y * 1.86 + 3), textcoords='data',
                     size=20,
                     bbox=dict(boxstyle="round4,pad=.5", fc="0.8"),
                     arrowprops=dict(arrowstyle="->",
                                     connectionstyle="angle,angleA=0,angleB=-90,rad=10"))

        plt.savefig(res_path + '{}_4_last_process.jpg'.format(self.path))

        series = pd.Series({'Thermal Decomposition Temperature': tem, 'Y1': y1
                               , 'Y2': final_point_y}, name=name)

        return series

    def forward(self, file_name, file_path, res_path, df):
        self.data_read(file_name, file_path)
        self.data_derivat_first()
        self.data_derivat_second()
        self.data_plot(res_path)
        series = self.data_tangent(res_path, df)
        return series


def defect_cal(df):
    defects = []
    for i in range(len(df)):
        y1 = df['Y1'][i]
        y2 = df['Y2'][i]
        defect = (1.86 - y1 / y2) / 0.86 * 100
        defect = float('%.2f' % defect)
        defects.append(defect)
    defects = pd.DataFrame(defects, columns=['Defcets'])
    Def_df = pd.concat([df, defects], axis=1)
    return Def_df


def main():

    files_path = 'test-data/'

    dirs_ = os.listdir(files_path)
    res_path = files_path + 'result/'
    Tem_df = pd.DataFrame(columns=['Name', 'Thermal Decomposition Temperature', 'Y1', 'Y2'])
    df = pd.DataFrame(columns=['Name', 'Thermal Decomposition Temperature', 'Y1', 'Y2'])
    if os.path.exists(res_path):
        pass
    else:
        os.mkdir(res_path)
    for i in dirs_:
        res = Data_process()
        if '.csv' in i:
            file_path = files_path + i
            series = res.forward(i, file_path, res_path, df=df)
            Tem_df = Tem_df.append(series)

    Tem_df.to_csv('Thermal Decomposition Temperature.csv')
    Def_df = defect_cal(Tem_df)
    Def_df.to_csv('Defects.csv')

    print('over ! ')


if __name__ == '__main__':
    main()
