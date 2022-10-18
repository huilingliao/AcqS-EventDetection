import torch
import numpy as np
from tabulate import tabulate
from scipy.spatial import ConvexHull


def ROI_lst_gen(dfn):
    if dfn == "forrester":
        n_dim = 1
        n_rois = 2
        threshold_value = -0.5
        ymin = -6.02074
        roi_bds_lst = [(0.1, 0.2), (0.6, 0.85)]
    elif dfn == "gramacylee":
        n_dim = 1
        n_rois = 6
        threshold_value = 0.
        ymin = -0.8690
        roi_bds_lst = [(0.5, 0.6), (0.7, 0.8), (0.9, 1.0), (1.1, 1.2), (1.3, 1.4), (1.5, 1.6)]
    elif dfn == "branin":
        n_dim = 2
        n_rois = 3
        threshold_value = 5.
        ymin = 0.
        roi_bds_lst = [[(-5., -1), (9., 15.)], [(2., 5.), (0., 6.)], [(8., 10.), (0., 6.)]]
    elif dfn == "bukin":
        n_dim = 2
        n_rois = None
        threshold_value = 10.
        ymin = 0.
        roi_bds_lst = None
    elif dfn == "cos25":
        n_dim = 2
        n_rois = 25
        threshold_value = 0.3
        ymin = 0.
        roi_bds_lst = []
        bds = [-5. * np.pi, -3. * np.pi, - np.pi, np.pi, 3. * np.pi, 5. * np.pi]
        for i in range(5):
            for j in range(5):
                roi_bds_lst.append([(bds[i], bds[i + 1]), (bds[j], bds[j + 1])])
    elif dfn == "sixhump":
        n_dim = 2
        n_rois = 2
        ymin = -1.0316
        threshold_value = -0.5
        roi_bds_lst = [[(-1., 1.), (0., 1.1)], [(-1., 1.), (-1.1, 0.)]]
        # roi_bds_lst = [[(-2., -1.), (0., 1.)], [(-1., 1.), (0., 1.1)], [(-1., 1.), (-1.1, 0.)], [(1., 2.), (-1., 0.)]]
    elif dfn == "michalewicz_2":
        n_dim = 2
        n_rois = 1
        threshold_value = -1.5
        ymin = -1.8013
        roi_bds_lst = [[(1.5, 2.5), (1., 2.)]]
    elif dfn == "ackley_6":
        n_dim = 6
        n_rois = None
        threshold_value = 10.
        ymin = 0.
        roi_bds_lst = None
    elif dfn == "michalewicz_5":
        n_dim = 5
        n_rois = None
        threshold_value = -3.5
        ymin = -4.6877
        roi_bds_lst = None
    elif dfn == "hartmann":
        n_dim = 6
        n_rois = None
        threshold_value = -2.8
        ymin = -3.32237
        roi_bds_lst = None
    elif dfn == "ackley_10":
        n_dim = 10
        n_rois = None
        threshold_value = 15.
        ymin = 0.
        roi_bds_lst = None
    elif dfn == "michalewicz_10":
        n_dim = 10
        n_rois = None
        threshold_value = -5.5
        ymin = -9.6602
        roi_bds_lst = None
    elif dfn == "xgboost":
        n_dim = 6
        n_rois = None
        threshold_value = -0.995
        ymin = -1.
        roi_bds_lst = None
    elif dfn == "cartpole":
        n_dim = 3
        n_rois = None
        threshold_value = -150
        ymin = -200
        roi_bds_lst = None
    elif dfn == "circ_sim":
        n_dim = 14
        n_rois = None
        threshold_value = -5.0
        ymin = -5.2
        roi_bds_lst = None

    return {'n_dim': n_dim, 'n_rois': n_rois, 'threshold_value': threshold_value, 'ymin': ymin, 'roi_lst': roi_bds_lst}


def check_roi(resi, n_rois, threshold, roi_bds_lst):
    budget = resi.shape[0]
    n_dim = resi.shape[1] - 1
    idx = np.where(resi[:, n_dim] < threshold)[0]
    n_roi_over_time = np.zeros((budget, ))
    if len(idx) == 0:
        n_founds = np.array([0.] * n_rois)
        n_roi_founds = 0.
        detect_roi_time = np.array([np.nan] * n_rois)
        area_found = 0.
    else:
        sel_x = resi[idx, :n_dim]
        detect_roi_time = np.array([np.nan] * n_rois)
        detect_roi_time_lst = []
        n_founds = np.array([0.] * n_rois)
        for i in range(n_rois):
            detect_lsti = []
            for j in range(sel_x.shape[0]):
                sel_xj = sel_x[j]
                if n_dim == 1:
                    if sel_xj[0] < roi_bds_lst[i][1] and sel_xj[0] > roi_bds_lst[i][0]:
                        n_founds[i] += 1
                        detect_lsti.append(j)
                        # detect_roi_time_lst[i].append(j)
                        if np.isnan(detect_roi_time[i]):
                            detect_roi_time[i] = idx[j] + 1
                else:
                    if all([sel_xj[k] < roi_bds_lst[i][k][1] and sel_xj[k] > roi_bds_lst[i][k][0] for k in range(n_dim)]):
                        n_founds[i] += 1
                        detect_lsti.append(j)
                        # detect_roi_time_lst[i].append(j)
                        if np.isnan(detect_roi_time[i]):
                            detect_roi_time[i] = idx[j] + 1
            detect_roi_time_lst.append(detect_lsti)
               
        n_roi_founds = sum(n_founds > 0)
        times_found = []
        areas_found = [0.] * n_roi_founds
        detect_roi_time.sort()
            
        if n_roi_founds == 1:
            n_roi_over_time[int(detect_roi_time[0]):] = 1.
        elif n_roi_founds > 1:
            for kk in range(n_roi_founds - 1):
                n_roi_over_time[int(detect_roi_time[kk]):int(detect_roi_time[kk + 1])] = kk + 1.
            n_roi_over_time[int(detect_roi_time[n_roi_founds - 1]):] = n_roi_founds

        area_found = 0.
        for i in range(n_rois):
            if detect_roi_time_lst[i] != []:
                n_x_in_rois = len(detect_roi_time_lst[i])
                x_in_roi = sel_x[detect_roi_time_lst[i]]
                if (n_dim > 1 and n_x_in_rois > 2):
                    hull = ConvexHull(x_in_roi)
                    area_found += hull.volume
                elif n_dim == 1 and n_x_in_rois > 1:
                    area_found += max(x_in_roi) - min(x_in_roi)

    return n_founds, n_roi_founds, detect_roi_time, area_found, n_roi_over_time


def eval_metrics_tab(dfn, res, kth_lst):
    n_reps = len(res)
    n_dim = res[0].shape[1] - 1
    if n_dim > 2:
        n_init = n_dim + 1
    else:
        n_init = 3
    if dfn == "circ_sim":
        n_init = 3

    budget = res[0].shape[0] - n_init

    # get rois/ymin/threshold for dfn
    info_dict = ROI_lst_gen(dfn)

    # average regrets over evaluations
    regrets = np.zeros((n_reps, budget + n_init))
    ymin = info_dict['ymin']
    for i in range(n_reps):
        regrets[i, :] = np.array([(res[i][:j, n_dim] - ymin).min() for j in range(1, budget + n_init + 1)])
    ave_regrets = regrets.mean(axis = 0)

    res = [resi[n_init:, :] for resi in res]
    if dfn in ['xgboost', 'cartpole', 'circ_sim']:
        for i in range(n_reps):
            res[i][:, n_dim] = - res[i][:, n_dim]

    # get the average of ymin found
    ave_ymin = np.mean([resi[:, n_dim].min() for resi in res])
    ymin_achieve = np.array([resi[:,n_dim].min() for resi in res]).min()

    # num of ROIs or num of failures found
    n_rois = info_dict['n_rois']
    threshold_val = info_dict['threshold_value']
    roi_bds_lst = info_dict['roi_lst']
    
    ave_num_failures = np.mean([sum(resi[:, n_dim] < threshold_val) for resi in res])
    detect_time_lst = [np.where(resi[:, n_dim] < threshold_val)[0] + 1 for resi in res]
    failure_detected = [resi[resi[:, n_dim] < threshold_val, :] for resi in res]
    ymin_over_time = np.zeros((n_reps, budget))
    for i in range(n_reps):
        resi = res[i]
        ymin_over_time[i, :] = np.array([resi[:j, n_dim].min() for j in range(1, 1 + budget)])
    ave_ymin_over_time = ymin_over_time.mean(axis = 0)

    if n_rois is None:
        num_fail_rois = ave_num_failures
        ratio_rois = np.nan
        detect_rois_time = np.zeros((n_reps, len(kth_lst)))
        n_roi_over_time = np.zeros((n_reps, budget))
        n_founds_mat = np.zeros((n_reps, len(kth_lst)))
        n_roi_founds = np.zeros((n_reps, ))
        area_founds = np.nan
        for i in range(n_reps):
            resi = res[i]
            detect_lst_i = detect_time_lst[i]
            idxk = [np.nan] * len(kth_lst)
            for kk in range(len(kth_lst)):
                if len(detect_lst_i) > kth_lst[kk] - 1:
                    detect_rois_time[i, kk] = detect_lst_i[kth_lst[kk] - 1] + 1
            # n_founds_mat[i, :] = 1. * (np.isnan(detect_rois_time[i, :]) == False)
            n_founds_mat[i, :] = 1. * (detect_rois_time[i, :] > 0)
            # roi over time
            n_detect_i = len(detect_lst_i)
            if n_detect_i == 1:
                n_roi_over_time[i, int(detect_lst_i[0]):] = 1.
            elif n_detect_i > 1:
                for kk in range(n_detect_i - 1):
                    n_roi_over_time[i, int(detect_lst_i[kk]):int(detect_lst_i[kk+1])] = kk + 1.
                n_roi_over_time[i, int(detect_lst_i[n_detect_i - 1]):] = n_detect_i    
    else:
        detect_rois_time = np.zeros((n_reps, n_rois))
        n_roi_over_time = np.zeros((n_reps, budget))
        n_founds_mat = np.zeros((n_reps, n_rois))
        n_roi_founds = np.zeros((n_reps, ))
        area_founds = np.zeros((n_reps, ))
        for i in range(n_reps):
            resi = res[i]
            n_found_i, n_roi_found, detect_roi_time_i, area_found_i, n_roi_over_time_i = check_roi(resi, n_rois, threshold_val, roi_bds_lst)
            n_roi_founds[i] = n_roi_found
            n_founds_mat[i, :] = n_found_i
            detect_rois_time[i, :] = detect_roi_time_i
            area_founds[i] = area_found_i
            n_roi_over_time[i,:] = n_roi_over_time_i
            
        num_fail_rois = np.mean(n_roi_founds)
        ratio_rois = num_fail_rois / n_rois
        area_founds = np.mean(area_founds)
    
    return {'ave_ymin': ave_ymin, 
            'ymin_achieve': ymin_achieve,
            'ymin_over_time': ave_ymin_over_time,
            'ave_regrets': ave_regrets, 
            'ave_num_failures': ave_num_failures, 
            'num_fail_rois': num_fail_rois, 
            'ratio_rois': ratio_rois, 
            'detect_rois_time': detect_rois_time, 
            'n_founds_mat': n_founds_mat, 
            'n_roi_founds': n_roi_founds, 
            'area_founds': area_founds,
            'detect_time_all': detect_time_lst,
            'failure_detected': failure_detected,
            'n_roi_over_time': n_roi_over_time}
    # return ave_ymin, ave_regrets, num_fail_rois, ratio_rois, detect_rois_time, n_founds_mat, n_roi_founds, area_founds




def summary_tab_all(res_lst, mdl_lst, dataset_lst, acq_lst, kth_lst):
    # results of tpe, random, opt and sample
    n_res = len(res_lst)

    # get the number of rows
    n_rows = 0
    col_names = []
    for mdl in mdl_lst:
        if mdl == "random":
            n_rows += 1
            col_names.append("random")
        elif mdl == "tpe":
            n_rows += 1
            col_names.append("tpe")
        elif mdl in ["opt", "sample"]:
            n_rows += len(acq_lst)
            col_names.extend([mdl + "_" + acq for acq in acq_lst])

    # get the number of metrics of interest
    tab_lst = []
    regrets_tab_lst = []
    n_roi_over_time_lst = []
    ymin_over_time_lst = []
    available_res_names_lst = []
    metrics_names_lst = []
    for dataset in dataset_lst:
        info_dict = ROI_lst_gen(dataset)
        if info_dict['n_rois'] is None:
            metrics_name_lst = ["t_" + str(i) for i in kth_lst]
            metrics_name_lst.extend(["r_n_" + str(i) for i in kth_lst])
        else: 
            metrics_name_lst = ["t_" + str(i) for i in range(1, 1 + info_dict['n_rois'])]
            metrics_name_lst.extend(["r_n_" + str(i) for i in range(1, 1 + info_dict['n_rois'])])
        # metrics_name_lst.extend(["n_failures", "n_rois_found", "Area/Length", "Ratio_Found", "ymin"])
        metrics_name_lst.extend(["N()", "N_r()", "Area()", "R_All", "ave_ymin"])
        n_cols = len(metrics_name_lst)
        tabi = []
        regrets_tab = []
        n_roi_over_time_tab = []
        ymin_over_time_tab = []
        available_res_names = []
        metrics_names_lst.append(metrics_name_lst)

        for i in range(n_res):
            mdl = mdl_lst[i]
            print("dataset = " + dataset + " | mdl =  " + mdl)
            if mdl in ["random", "tpe"]:
                acq = ""
                if res_lst[i][dataset] is None:
                    tabii = [mdl] + [np.nan] * n_cols
                else:
                    available_res_names.append(mdl)
                    detect_ts = res_lst[i][dataset]['detect_rois_time']
                    n_reps = detect_ts.shape[0]
                    n_cols = detect_ts.shape[1]
                    ave_ts = [round(np.mean(detect_ts[detect_ts[:, p] > 0, p]), 2) for p in range(n_cols)]
                    # ave_ts = np.nanmean(detect_ts, axis = 0)
                    # sd_ts = np.nanstd(detect_ts, axis = 0)
                    # ave_sd_ts = [str(round(ave_ts[p], 2)) + "(" + str(round(sd_ts[p], 2)) + ")" for p in range(detect_ts.shape[1])]
                
                    # n_founds_mat = res_lst[i][dataset]['n_founds_mat']
                    # ave_ns = n_founds_mat.mean(axis = 0)
                    ## sd_ns = n_founds_mat.std(axis = 0)
                    ## ave_sd_ns = [str(round(ave_ns[p], 2)) + "(" + str(round(sd_ns[p], 2)) + ")" for p in range(detect_ts.shape[1])]

                    if info_dict['n_rois'] is None:
                        ave_nna = [round(sum(detect_ts[:, j] > 1.)/n_reps, 2) for j in range(len(kth_lst))]
                    else:
                        ave_nna = [round(sum(np.isnan(detect_ts)[:, j] == False)/n_reps, 2) for j in range(info_dict['n_rois'])]
                    # ave_nna_ns = [str(round(ave_ns[p], 2)) + "(" + str(round(ave_nna[p], 2)) + ")" for p in range(detect_ts.shape[1])]
                    ave_fail = round(res_lst[i][dataset]['ave_num_failures'], 2)
                    ave_rois = round(res_lst[i][dataset]['num_fail_rois'], 2)
                    ave_area = round(res_lst[i][dataset]['area_founds'], 2)
                    ymin_got = round(res_lst[i][dataset]['ymin_achieve'], 2)
                    if info_dict['n_rois'] is None:
                        ave_rate = np.nan
                    else:
                        ave_rate = round(sum(res_lst[i][dataset]['n_roi_founds'] == info_dict['n_rois']) / n_reps, 2)
                    ave_ymin = round(res_lst[i][dataset]['ave_ymin'], 2)
                    # tabii = [mdl] + ave_sd_ts + ave_nna_ns + [ave_fail, ave_rois, ave_area, ave_rate, ave_ymin]
                    # tabii = [mdl] + ave_sd_ts + [ave_fail, ave_rois, ave_area, ave_rate, ave_ymin]
                    tabii = [mdl] + [acq] +  ave_ts + [round(ave_nna[p], 2) for p in range(len(ave_nna))]  + [ave_fail, ave_rois, ave_area, ave_rate, ave_ymin]
                    tabi.append(tabii)
                    regrets_tab.append(res_lst[i][dataset]['ave_regrets'])
                    n_roi_over_time_tab.append(res_lst[i][dataset]['n_roi_over_time'])
                    ymin_over_time_tab.append(res_lst[i][dataset]['ymin_over_time'])
            elif mdl in ["opt", "sample"]:
                for acq in acq_lst:
                    if res_lst[i][acq][dataset] is None:
                        tabii = [mdl] + [acq] + [np.nan] * n_cols
                    else:
                        available_res_names.append(mdl + "_" + acq)
                        detect_ts = res_lst[i][acq][dataset]['detect_rois_time']
                        n_reps = detect_ts.shape[0]
                        n_cols = detect_ts.shape[1]
                        ave_ts = [round(np.mean(detect_ts[detect_ts[:, p] > 0, p]), 2) for p in range(n_cols)]
                        # ave_ts = np.nanmean(detect_ts, axis = 0)
                        # sd_ts = np.nanstd(detect_ts, axis = 0)
                        # ave_sd_ts = [str(round(ave_ts[p], 2)) + "(" + str(round(sd_ts[p], 2)) + ")" for p in range(detect_ts.shape[1])]

                        # n_founds_mat = res_lst[i][acq][dataset]['n_founds_mat']
                        # ave_ns = n_founds_mat.mean(axis = 0)
                        ## sd_ns = n_founds_mat.std(axis = 0)
                        ## ave_sd_ns = [str(round(ave_ns[p], 2)) + "(" + str(round(sd_ns[p], 2)) + ")" for p in range(detect_ts.shape[1])]

                        if info_dict['n_rois'] is None:
                            ave_nna = [sum(detect_ts[:, j] > 0)/n_reps for j in range(len(kth_lst))]
                        else:
                            ave_nna = [round(sum(np.isnan(detect_ts)[:, j] == False)/n_reps, 2) for j in range(info_dict['n_rois'])]
                        # ave_nna_ns = [str(round(ave_ns[p], 2)) + "(" + str(round(ave_nna[p], 2)) + ")" for p in range(detect_ts.shape[1])]
                        ave_fail = round(res_lst[i][acq][dataset]['ave_num_failures'], 2)
                        ave_rois = round(res_lst[i][acq][dataset]['num_fail_rois'], 2)
                        ave_area = round(res_lst[i][acq][dataset]['area_founds'], 2)
                        ymin_got = round(res_lst[i][acq][dataset]['ymin_achieve'], 2)
                        if info_dict['n_rois'] is None:
                            ave_rate = np.nan
                        else:
                            ave_rate = round(sum(res_lst[i][acq][dataset]['n_roi_founds'] == info_dict['n_rois']) / n_reps, 2)
                        ave_ymin = round(res_lst[i][acq][dataset]['ave_ymin'], 2)
                        # tabii = [mdl + "_" + acq] + ave_sd_ts + ave_nna_ns + [ave_fail, ave_rois, ave_area, ave_rate, ave_ymin]
                        # tabii = [mdl + "_" + acq] + ave_sd_ts + [ave_fail, ave_rois, ave_area, ave_rate, ave_ymin]
                        tabii = [mdl] + [acq] + ave_ts + [round(ave_nna[p], 2) for p in range(len(ave_nna))]  + [ave_fail, ave_rois, ave_area, ave_rate, ave_ymin]
                        tabi.append(tabii)
                        regrets_tab.append(res_lst[i][acq][dataset]['ave_regrets'])
                        n_roi_over_time_tab.append(res_lst[i][acq][dataset]['n_roi_over_time'])
                        ymin_over_time_tab.append(res_lst[i][acq][dataset]['ymin_over_time'])
        # append to the tab_lst all
        tab_lst.append(tabi)
        regrets_tab_lst.append(regrets_tab)
        n_roi_over_time_lst.append(n_roi_over_time_tab)
        ymin_over_time_lst.append(ymin_over_time_tab)
        available_res_names_lst.append(available_res_names)

    return tab_lst, regrets_tab_lst, n_roi_over_time_lst, ymin_over_time_lst, available_res_names_lst, metrics_names_lst

