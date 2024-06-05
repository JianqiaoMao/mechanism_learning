#%%
import causalBootstrapping as cb
import numpy as np
#%%
def mechanism_classifier(cause_data, mediator_data, effect_data, dist_map, ml_model, rebalance = False, n_samples = None, cb_mode = "fast"):
    cause_var_name = list(cause_data.keys())[0]
    effect_var_name = list(effect_data.keys())[0]
    if rebalance:
        cause_unique = np.unique(cause_data[cause_var_name])
        if n_samples is None:
            N = cause_data[cause_var_name].shape[0]
            cause_unique_n = len(cause_unique)
            n_samples = [int(N/cause_unique_n)]*cause_unique_n
        cb_data = {}
        for i, interv_value in enumerate(cause_unique):
            cb_data_simu = cb.frontdoor_simu(cause_data = cause_data,
                                             mediator_data = mediator_data,
                                             effect_data = effect_data,
                                             dist_map = dist_map,
                                             mode = cb_mode,
                                             n_samples = n_samples[i],
                                             interv_value = interv_value)
            for key in cb_data_simu:
                if i == 0:
                    cb_data[key] = cb_data_simu[key]
                else:
                    cb_data[key] = np.vstack((cb_data[key], cb_data_simu[key]))
    else:
        cb_data = cb.frontdoor_simple(cause_data = cause_data,
                                      mediator_data = mediator_data,
                                      effect_data = effect_data,
                                      dist_map = dist_map,
                                      mode = cb_mode)
    ml_model = ml_model.fit(cb_data[effect_var_name], cb_data["intv_"+cause_var_name].ravel())
    
    return ml_model