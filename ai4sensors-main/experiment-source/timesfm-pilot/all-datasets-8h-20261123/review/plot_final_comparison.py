"""Write a portable high-resolution figure of the delivered, full-precision per-domain gains."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,PercentFormatter


def main():
    here=Path(__file__).resolve().parent
    with (here/'report-addendum.json').open(encoding='utf8')as f:
        rows=json.load(f)['queries']['all8-final-cross-source']['rows']
    with (here.parent/'selection-lock.json').open(encoding='utf8')as f: batch=json.load(f)['selected']['batch']
    assert len(rows)==12
    with plt.rc_context({'font.family':['Microsoft YaHei'],'axes.unicode_minus':False,'font.size':11}):
        fig,ax=plt.subplots(figsize=(9.0,6.4),layout='constrained')
        vals=[r['gain_fraction']for r in rows if r['gain_fraction'] is not None]
        low,high=min([0.,*vals]),max([0.,*vals])
        width=max(high-low,.01);offset=.013*width
        ax.set_xlim(low-.23*width,high+.24*width)
        ax.axvline(0,color='#405369',linewidth=1)
        for i,row in enumerate(rows):
            v=row['gain_fraction']
            if v is None:
                ax.text(offset,i,'缺分',va='center',color='#5c6571')
            else:
                ax.barh(i,v,height=.55,color='#28796c' if v>=0 else '#ad5638',zorder=2)
                ax.text(v+(offset if v>=0 else -offset),i,f'{v:+.2%}',ha='left' if v>=0 else 'right',va='center',color='#23313a',fontsize=10)
        ax.axhline(7.5,color='#b0b8be',linewidth=.8,linestyle='--')
        ax.set_yticks(range(len(rows)),[r['c0'] for r in rows]);ax.invert_yaxis()
        ax.tick_params(left=False,bottom=False,labelcolor='#293840')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7));ax.xaxis.set_major_formatter(PercentFormatter(1,decimals=0))
        ax.grid(axis='x',color='#dce3e6',linewidth=.7,zorder=0)
        ax.spines[['top','right','left','bottom']].set_visible(False)
        ax.set_xlabel('相对默认 FullVQF 的倾角误差改善，正值表示误差下降',labelpad=12)
        ax.set_title(f'{batch} 冻结后在各数据集上的表现',loc='left',fontsize=19,pad=20,color='#18252b')
        fig.text(.02,-.035,'每条记录先计算倾角 RMSE，再在数据集内等权平均。虚线上方为八套主评，下方为四套补充迁移检验。',fontsize=9,color='#495963')
        fig.savefig(here/'final-domain-gains.png',dpi=170,bbox_inches='tight',facecolor='white')
        plt.close(fig)


if __name__=='__main__':main()
