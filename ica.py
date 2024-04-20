import numpy as np

def f1(x):
	return 1/(1 + np.exp(-x))

def f1_der(x):
    d = f1(x)
    return d*(1 - d)

def center_data(X):
    data = []
    for result in X:
        centered = result - np.mean(result)
        centered /= 32768
        data.append(centered)
    return data

def whiten(X):
    covar_of_sig = np.cov(X)
    eig_val, eig_vec = np.linalg.eig(covar_of_sig)
    diag_eig = np.diag(eig_val)
    inv_sqr_of_diag = np.sqrt(np.linalg.pinv((diag_eig)))
    whiten_trans = np.dot(eig_vec, np.dot(inv_sqr_of_diag, eig_vec.T))
    whitened_sig = np.dot(whiten_trans, X)
    return whitened_sig

def ica(audio_sources, epsilon = 1e7):
    centered = center_data(audio_sources)
    sig_matrix = np.vstack(centered)
    whitened_signal_matr = whiten(sig_matrix)

    comps_of_V = []
    for i in range(whitened_signal_matr.shape[0]):
        num_of_sources = sig_matrix.shape[0]
        length_of_track = sig_matrix.shape[1]

        v1 = np.random.rand(num_of_sources)
        v1 = v1/np.linalg.norm(v1)
        v2 = np.random.rand(num_of_sources)
        v2 = v2/np.linalg.norm(v2)

        while( (1 - np.abs(np.dot(v1.T,v2))) > epsilon):
            v1 = v2
            first = np.dot(whitened_signal_matr, f1(np.dot(v2.T, whitened_signal_matr)))/length_of_track
            second = np.mean(f1_der(np.dot(v2.T, whitened_signal_matr)))*v2
            v2 = first - second
            v3 = v2
            for pres_comp in comps_of_V:
                v3 = v3 - np.dot(v2.T, pres_comp)*pres_comp
            v2 = v3
            v2 = v2/np.linalg.norm(v2)
        comps_of_V.append(v1)
    V = np.vstack(comps_of_V)
    S = np.dot(V, whitened_signal_matr)
    return S
