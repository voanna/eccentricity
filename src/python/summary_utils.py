import itertools
from functools import lru_cache
import numpy as np
from run_crowding_acc import edge, step, target_eccs, flanker_eccs, all_jobs, EXPERIMENT, best_models
all_models = best_models(run_from_expt_dir=True)
import os
import matplotlib.pyplot as plt

@lru_cache(maxsize=None)
def get_no_flanker_precision(precision_dir):
    global edge, step, target_eccs, flanker_eccs, all_jobs
    no_flanker_precision = np.ones((len(target_eccs), len(flanker_eccs)))*np.nan
    for i, te in enumerate(target_eccs):
        for j, fe in enumerate(flanker_eccs):
            job_id = [index for index, job in enumerate(all_jobs) if job[0] == 0 and job[1] == te and job[2] == fe]
            try:
                no_flanker_precision[i, j] = np.genfromtxt(os.path.join(precision_dir, str(job_id[0]) + '.out'))
            except (OSError, IndexError):
                pass
    return no_flanker_precision

@lru_cache(maxsize=None)
def get_one_flanker_precision(precision_dir):
    global edge, step, target_eccs, flanker_eccs, all_jobs
    one_flanker_precision = np.ones((len(target_eccs), len(flanker_eccs)))*np.nan
    for i, te in enumerate(target_eccs):
        for j, fe in enumerate(flanker_eccs):
            job_id = [index for index, job in enumerate(all_jobs) if job[0] == 1 and job[1] == te and job[2] == fe]
            try:
                one_flanker_precision[i, j] = np.genfromtxt(os.path.join(precision_dir, str(job_id[0]) + '.out'))
            except (OSError, IndexError):
                pass
    return one_flanker_precision

@lru_cache(maxsize=None)
def get_two_flanker_precision(precision_dir):
    global edge, step, target_eccs, flanker_eccs, all_jobs
    two_flanker_precision = np.ones((len(target_eccs), len(flanker_eccs)))*np.nan
    for i, te in enumerate(target_eccs):
        for j, fe in enumerate(flanker_eccs):
            job_id = [index for index, job in enumerate(all_jobs) if job[0] == 2 and job[1] == te and job[2] == fe]
            try:
                two_flanker_precision[i, j] = np.genfromtxt(os.path.join(precision_dir, str(job_id[0]) + '.out'))
            except (OSError, IndexError):
                pass
    return two_flanker_precision



def spacing_120(model_name, return_handles=False):
    precision_dir = 'gen/precision_{}/'.format(model_name)
    global edge, step, target_eccs, flanker_eccs, all_jobs    
    no_flanker_precision = get_no_flanker_precision(precision_dir)
    one_flanker_precision = get_one_flanker_precision(precision_dir)
    two_flanker_precision = get_two_flanker_precision(precision_dir)
    
    fig = plt.figure()

    spacing = 120
    ax_eccs = target_eccs + spacing
    ax_x_idx, ax_y_idx = [], []
    for i, eccs  in enumerate(zip(ax_eccs, target_eccs)):

        f, t = eccs
        f_idx = np.where(flanker_eccs == f)
        t_idx = np.where(target_eccs == t)
        try:
            ax_y_idx.append(f_idx[0][0])
            ax_x_idx.append(t_idx[0][0])
        except IndexError:
            pass

    xa_eccs = target_eccs - spacing
    xa_x_idx, xa_y_idx = [], []
    for i, eccs  in enumerate(zip(xa_eccs, target_eccs)):
        f, t = eccs
        f_idx = np.where(flanker_eccs == f)
        t_idx = np.where(target_eccs == t)
        try:
            xa_y_idx.append(f_idx[0][0])
            xa_x_idx.append(t_idx[0][0])
        except IndexError:
            pass
    xaxis = target_eccs
    plt.plot(xaxis[ax_x_idx], no_flanker_precision[ax_x_idx, ax_y_idx], '-', color='#003310', lw=8, ms=30)
    plt.plot(xaxis[xa_x_idx], one_flanker_precision[xa_x_idx, xa_y_idx], '-', marker='>', color='#99ffb9',  markeredgewidth=2, lw=8, ms=30)
    plt.plot(xaxis[ax_x_idx], one_flanker_precision[ax_x_idx, ax_y_idx], '-', marker='<', color='#00e649',  markeredgewidth=2, lw=8, ms=30)
    plt.plot(xaxis[ax_x_idx], two_flanker_precision[ax_x_idx, ax_y_idx], '-', marker='s', color='#008028',  markeredgewidth=2, lw=8, ms=30)
    plt.plot(xaxis, [0.2]*len(xaxis), ':', color='0.3', lw=8)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)

    plt.ylim([0,1])
    plt.ylabel('Accuracy', fontsize=75)
    plt.xlabel('Target eccentricity', fontsize=75)
    plt.xlim([np.min(target_eccs), np.max(target_eccs)])

    plt.legend(['a', 'xa', 'ax', 'xax', 'chance'], loc='lower center', ncol=5, fontsize=35, framealpha=0.0, numpoints=1, handlelength=1)

    if return_handles:
        return fig


def spacing_240(model_name, return_handles=False):

    precision_dir = 'gen/precision_{}/'.format(model_name)
    global edge, step, target_eccs, flanker_eccs, all_jobs    
    no_flanker_precision = get_no_flanker_precision(precision_dir)
    one_flanker_precision = get_one_flanker_precision(precision_dir)
    two_flanker_precision = get_two_flanker_precision(precision_dir)
    
    fig = plt.figure()

    # since in the case of no flankers, the results are the same for all spacings.
    # due to  ??? 
    # Figure out if this produces a hole in the plot
    # spacing = 120
    # ax_eccs = target_eccs + spacing
    # ax_x_idx, ax_y_idx = [], []
    # for i, eccs  in enumerate(zip(ax_eccs, target_eccs)):

    #     f, t = eccs
    #     f_idx = np.where(flanker_eccs == f)
    #     t_idx = np.where(target_eccs == t)
    #     try:
    #         ax_y_idx.append(f_idx[0][0])
    #         ax_x_idx.append(t_idx[0][0])
    #     except IndexError:
    #         pass

    # plt.plot(xaxis[ax_x_idx], no_flanker_precision[ax_x_idx, ax_y_idx], '-', color='#003310', lw=8, ms=30)

    spacing = 240
    ax_eccs = target_eccs + spacing
    ax_x_idx, ax_y_idx = [], []
    for i, eccs  in enumerate(zip(ax_eccs, target_eccs)):

        f, t = eccs
        f_idx = np.where(flanker_eccs == f)
        t_idx = np.where(target_eccs == t)
        try:
            ax_y_idx.append(f_idx[0][0])
            ax_x_idx.append(t_idx[0][0])
        except IndexError:
            pass

    xa_eccs = target_eccs - spacing
    xa_x_idx, xa_y_idx = [], []
    for i, eccs  in enumerate(zip(xa_eccs, target_eccs)):
        f, t = eccs
        f_idx = np.where(flanker_eccs == f)
        t_idx = np.where(target_eccs == t)
        try:
            xa_y_idx.append(f_idx[0][0])
            xa_x_idx.append(t_idx[0][0])
        except IndexError:
            pass

    xaxis = target_eccs
    plt.plot(xaxis[ax_x_idx], no_flanker_precision[ax_x_idx, ax_y_idx], '-', color='#003310', lw=8, ms=30)

    plt.plot(xaxis[xa_x_idx], one_flanker_precision[xa_x_idx, xa_y_idx], '-', marker='>', color='#99ffb9',  markeredgewidth=2, lw=8, ms=30)
    plt.plot(xaxis[ax_x_idx], one_flanker_precision[ax_x_idx, ax_y_idx], '-', marker='<', color='#00e649',  markeredgewidth=2, lw=8, ms=30)
    plt.plot(xaxis[ax_x_idx], two_flanker_precision[ax_x_idx, ax_y_idx], '-', marker='s', color='#008028',  markeredgewidth=2, lw=8, ms=30)
    plt.plot(xaxis, [0.2]*len(xaxis), ':', color='0.3', lw=8)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)

    plt.ylim([0,1])
    plt.ylabel('Accuracy', fontsize=75)
    plt.xlabel('Target eccentricity', fontsize=75)
    plt.xlim([np.min(target_eccs), np.max(target_eccs)])
    plt.legend(['a', 'xa', 'ax', 'xax', 'chance'], loc='lower center', ncol=5, fontsize=35, framealpha=0.0, numpoints=1, handlelength=1, labelspacing=0.1)

    if return_handles:
        return fig


def target_fixed_at_0(model_name, return_handles=False):
    precision_dir = 'gen/precision_{}/'.format(model_name)
    global edge, step, target_eccs, flanker_eccs, all_jobs

    no_flanker_precision = get_no_flanker_precision(precision_dir)
    one_flanker_precision = get_one_flanker_precision(precision_dir)
    two_flanker_precision = get_two_flanker_precision(precision_dir)
    

    fig = plt.figure()

    target_ecc = 0

    j = np.where(target_eccs == target_ecc)[0][0]
    nfp = no_flanker_precision[j, 0]

    plt.plot(flanker_eccs, nfp * np.ones(len(flanker_eccs)), '-', color ='#330033', lw=8, ms=30)
    plt.plot(flanker_eccs - target_ecc, one_flanker_precision[j, :], marker='<', color='#e600e6', markeredgewidth=2, lw=8, ms=30)
    plt.plot(flanker_eccs - target_ecc, two_flanker_precision[j, :], marker='s', color='#800080', markeredgewidth=2, lw=8, ms=30)
    plt.plot(flanker_eccs, 0.2 * np.ones(len(flanker_eccs)), ':', color='0.3', lw=8)
    plt.ylim([0,1])
    plt.xlim([0, np.max(flanker_eccs)])
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)

    plt.ylabel('Accuracy', fontsize=75)
    plt.xlabel('Flanker Eccentricity', fontsize=75)
    plt.legend(['a', 'ax', 'xax', 'chance'], loc='lower center', ncol=4, fontsize=35, framealpha=0.0, numpoints=1, handlelength=1)
    if return_handles:
        return fig

def target_fixed_at_720(model_name, return_handles=False):

    precision_dir = 'gen/precision_{}/'.format(model_name)
    global edge, step, target_eccs, flanker_eccs, all_jobs

    no_flanker_precision = get_no_flanker_precision(precision_dir)
    one_flanker_precision = get_one_flanker_precision(precision_dir)
    two_flanker_precision = get_two_flanker_precision(precision_dir)
    

    fig = plt.figure()
    target_ecc = 720
    j = np.where(target_eccs == target_ecc)[0][0]

    nfp = no_flanker_precision[j, 0] #no flanker precision

    plt.plot(flanker_eccs, nfp * np.ones(len(flanker_eccs)), '-', color ='#330033', lw=8, ms=30)
    plt.plot(target_ecc - flanker_eccs, one_flanker_precision[j, :], marker='>', color='#ff99ff', markeredgewidth=2, lw=8, ms=30)
    plt.plot(flanker_eccs, 0.2 * np.ones(len(flanker_eccs)), ':', color='0.3', lw=8)
    plt.xlim([0, np.max(flanker_eccs)])
    plt.xlim([0, np.max(flanker_eccs)])
    plt.ylim([0, 1])
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)

    plt.ylabel('Accuracy', fontsize=75)
    plt.xlabel('Flanker Eccentricity', fontsize=75)
    plt.legend(['a', 'xa', 'chance'], loc='lower center', ncol=3, fontsize=35, framealpha=0.0, numpoints=1, handlelength=1)
    if return_handles:
        return fig
