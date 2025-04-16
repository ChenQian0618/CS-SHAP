import matplotlib as mpl
import matplotlib.transforms as mtransforms
import seaborn as sns
import numpy as np

def setseaborn():
    # set color
    current_cmap = sns.color_palette("deep")
    sns.set(style="whitegrid")
    sns.set(style="ticks", context="notebook", font='Times New Roman', palette=current_cmap, font_scale=1)
    return current_cmap

def setdefault(ratio=1, fontsize=4, TrueType=True, font=['Arial']): # font=['Arial', 'simsun']
    '''
    mpl.rcParams: https://matplotlib.org/stable/users/explain/customizing.html
    {k:v for k,v in mpl.rcParams.items() if 'pad' in k}
    '''

    _ = setseaborn()
    # '#7EA4D1', '#807C7D', '#C1565E', '#DCA96A', '#82AD7F', '#79438E'
    # '#1b67ab', '#807C7D', '#C1565E', '#DCA96A', '#82AD7F', '#79438E'
    custom_colors = ['#1b67ab', '#C1565E', '#DCA96A', '#9B59B6', '#82AD7F', '#00788C', '#726A95',
                     '#20B2AA', '#839788']
    mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=custom_colors)

    mpl.rcParams.update({'figure.dpi': 600 / ratio,
                         'figure.figsize': [7 / 2.54 * ratio, 4 / 2.54 * ratio], })

    mpl.rcParams['axes.linewidth'] = 0.5 * ratio
    mpl.rcParams['lines.linewidth'] = 0.5 * ratio

    mpl.rcParams['font.family'] = font  # 'Times New Roman' 'Arial'
    mpl.rcParams['svg.fonttype'] = 'none'
    mpl.rcParams['pdf.fonttype'] = 42 if TrueType else 3  # 42: TrueType, 3: Type 3

    mpl.rcParams['font.size'] = fontsize
    mpl.rcParams.update({'xtick.labelsize': mpl.rcParams['font.size'],
                         'ytick.labelsize': mpl.rcParams['font.size'],
                         'axes.labelsize': mpl.rcParams['font.size'],
                         'axes.titlesize': mpl.rcParams['font.size'],
                         'legend.fontsize': mpl.rcParams['font.size'],
                         'legend.title_fontsize': mpl.rcParams['font.size'],
                         'figure.titlesize': mpl.rcParams['font.size'],
                         'figure.labelsize': mpl.rcParams['font.size'], })

    mpl.rcParams.update({'figure.subplot.bottom': 0.05,
                         'figure.subplot.hspace': 0.4,
                         'figure.subplot.left': 0.05,
                         'figure.subplot.right': 0.95,
                         'figure.subplot.top': 0.92,
                         'figure.subplot.wspace': 0.4, })

    mpl.rcParams.update({'xtick.direction': 'in',
                         'xtick.major.width': 0.5 * ratio,
                         'xtick.major.size': 2 * ratio,
                         'xtick.major.pad': 2.5 * ratio,
                         'xtick.major.top': False,
                         'ytick.direction': 'in',
                         'ytick.major.width': 0.5 * ratio,
                         'ytick.major.size': 2 * ratio,
                         'ytick.major.pad': 2.5 * ratio,
                         'ytick.major.right': False, })

    mpl.rcParams.update({'grid.color': '0.85',
                         'grid.alpha': 1,
                         'grid.linewidth': 0.3 * ratio,
                         'grid.linestyle': '--',
                         'legend.facecolor': 'none',
                         'figure.facecolor': 'none',
                         'axes.facecolor': 'none',
                         })

    mpl.rcParams.update({'axes.labelpad': 1 * ratio,
                         'axes.titlepad': 0.5 * ratio,
                         'axes.axisbelow': 'line', })  # 'line', 'true', 'false'

    mpl.rcParams.update({'hatch.linewidth': 0.25,
                         'hatch.color': 'b'})

    # mpl.rcParams['text.usetex'] = True
    mpl.rcParams["mathtext.fontset"] = 'cm'
    mpl.pyplot.switch_backend('tkagg')  # 'tkagg' 'Qt5Agg' 'pgf'

def ticklabel_format(ax,format='%g',which='both'):
    '''
    '''
    try:
        if which in ['both','x']:
            ax.xaxis.set_major_formatter(mpl.ticker.FormatStrFormatter(format))
        if which in ['both','y']:
            ax.yaxis.set_major_formatter(mpl.ticker.FormatStrFormatter(format))
    except:
        print('ticklabel_format error')

if __name__ == '__main__':
    setdefault()
