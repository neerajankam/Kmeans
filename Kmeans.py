import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.stats import multivariate_normal


def logexp(Z):
    """ Compute log(\sum_i exp(Z_i)) for some array Z."""
    return np.max(Z) + np.log(np.sum(np.exp(Z - np.max(Z))))


def log_lklihood(data, wts, means, covs):
    """ Compute the log likelihood of the data for a Gaussian mixture model with the given parameters. """
    no_clusters= len(means)
    no_dim = len(data[0])

    ll = 0
    for d in data:

        Z = np.zeros(no_clusters)
        for k in range(no_clusters):
            # Compute (x-mu)^T * Sigma^{-1} * (x-mu)
            delta = np.array(d) - means[k]
            exp_term = np.dot(delta.T, np.dot(np.linalg.inv(covs[k]), delta))

            # Compute log_lklihood contribution for this data point and this cluster
            Z[k] += np.log(wts[k])
            Z[k] -= 1 / 2. * (no_dim * np.log(2 * np.pi) + np.log(np.linalg.det(covs[k])) + exp_term)

        # Increment log_lklihood contribution of this data point across all clusters
        ll += logexp(Z)

    return ll


def comp_resp(data, wts, means, covs):
    '''E-step: compute responsibilities, given the current parameters'''
    num_data = len(data)
    no_clusters= len(means)
    resp = np.zeros((num_data, no_clusters))

    # Update resp matrix so that resp[i,k] is the responsibility of cluster k for data point i.

    for i in range(num_data):
        for k in range(no_clusters):
            resp[i, k] = wts[k] * multivariate_normal.pdf(data[i], mean=means[k], cov=covs[k])

    # Add up responsibilities over each data point and normalize
    row_sums = resp.sum(axis=1)[:, np.newaxis]
    resp = resp / row_sums

    return resp
def comp_softcounts(resp):
    # Compute the total responsibility assigned to each cluster, which will be useful when
    # implementing M-steps below. In the lectures this is called N^{soft}
    counts = np.sum(resp, axis=0)
    return counts


def comp_wts(counts):
    no_clusters= len(counts)
    wts = [0.] * no_clusters

    for k in range(no_clusters):
        # Update the weight for cluster k using the M-step update rule for the cluster weight, \hat{\pi}_k.

        wts[k] = counts[k] / np.sum(counts)

    return wts


def comp_means(data, resp, counts):
    no_clusters= len(counts)
    num_data = len(data)
    means = [np.zeros(len(data[0]))] * no_clusters

    for k in range(no_clusters):
        # Update means for cluster k using the M-step update rule for the mean variables.
        # This will assign the variable means[k] to be our estimate for \hat{\mu}_k.
        weighted_sum = 0.
        for i in range(num_data):
            weighted_sum += resp[i, k] * data[i]
        means[k] = weighted_sum / counts[k]

    return means


def comp_covs(data, resp, counts, means):
    no_clusters= len(counts)
    no_dim = len(data[0])
    num_data = len(data)
    covs = [np.zeros((no_dim, no_dim))] * no_clusters

    for k in range(no_clusters):
        # Update covs for cluster k using the M-step update rule for covariance variables.
        # This will assign the variable covs[k] to be the estimate for \hat{\Sigma}_k.
        weighted_sum = np.zeros((no_dim, no_dim))
        for i in range(num_data):
            weighted_sum += resp[i, k] * np.outer(data[i] - means[k], data[i] - means[k])
        covs[k] = weighted_sum / counts[k]

    return covs


def EM(data, init_means, init_covs, init_wts, iter_max=1000, thresh=1e-4):
    # Make copies of initial parameters, which we will update during each iteration
    means = init_means[:]
    covs = init_covs[:]
    wts = init_wts[:]

    # Infer dimensions of dataset and the number of clusters
    num_data = len(data)
    no_dim = len(data[0])
    no_clusters= len(means)

    # Initialize some variables
    resp = np.zeros((num_data, no_clusters))
    ll = log_lklihood(data, wts, means, covs)
    ll_trace = [ll]

    for it in range(iter_max):

        #Compute responsibilities
        resp = comp_resp(data, wts, means, covs)

        # Compute the total responsibility assigned to each cluster, which will be useful when
        # implementing M-steps below. In the lectures this is called N^{soft}
        counts = comp_softcounts(resp)

        # Update the weight for cluster k using the M-step update rule for the cluster weight, \hat{\pi}_k.
        wts = comp_wts(counts)

        # Update means for cluster k using the M-step update rule for the mean variables.
        # This will assign the variable means[k] to be our estimate for \hat{\mu}_k.
        means = comp_means(data, resp, counts)

        # Update covs for cluster k using the M-step update rule for covariance variables.
        # This will assign the variable covs[k] to be the estimate for \hat{\Sigma}_k.
        covs = comp_covs(data, resp, counts, means)

        # Compute the log_lklihood at this iteration
        ll_latest = log_lklihood(data, wts, means, covs)
        ll_trace.append(ll_latest)

        # Check for convergence in log-likelihood and store
        if (ll_latest - ll) < thresh and ll_latest > -np.inf:
            break
        ll = ll_latest

    if it % 5 != 0:
        print("Iteration %s" % it)

    out = {'wts': wts, 'means': means, 'covs': covs, 'loglik': ll_trace, 'resp': resp}

    return out

def generate_MoG_data(num_data, means, covs, wts):
    """ Creates a list of data points """
    no_clusters= len(wts)
    data = []
    for i in range(num_data):
        #  Use np.random.choice and wts to pick a cluster id greater than or equal to 0 and less than no_clusters.
        k = np.random.choice(len(wts), 1, p=wts)[0]

        # Use np.random.multivariate_normal to create data from this cluster
        x = np.random.multivariate_normal(means[k], covs[k])

        data.append(x)
    return data

# Model parameters
init_means = [
    [31, 60], # mean of cluster 1
    [20, 35], # mean of cluster 2
    [50, 48]  # mean of cluster 3
]
init_covs = [
   [[103.96243033,    3.34190299],
    [3.34190299,   96.1837673]], # covariance of cluster 1
     [[ 109.48397634,   -0.31492123],
 [  -0.31492123,  106.60518593]],# covariance of cluster 2
[[103.25448164, - 1.92097408],
 [-1.92097408,   93.93524127]]# covariance of cluster 3
]
init_wts = [2298/float(6000), 1904/float(6000), 1798/float(6000)]  # wts of each cluster

# Generate data
np.random.seed(4)
data = generate_MoG_data(100, init_means, init_covs, init_wts)


plt.figure()
d = np.vstack(data)
plt.plot(d[:,0], d[:,1],'ko')
plt.rcParams.update({'font.size':16})
plt.tight_layout()
plt.show()

np.random.seed(4)

# Initialization of parameters
chosen = np.random.choice(len(data), 3, replace=False)
initial_means = [data[x] for x in chosen]
initial_covs = [np.cov(data, rowvar=0)] * 3
initial_wts = [1/3.] * 3

# Run EM
results = EM(data, initial_means, initial_covs, initial_wts)

print(results['wts'])
print(results['means'])
print(results['covs'])

import matplotlib.mlab as mlab
def plot_contours(data, means, covs, title):
    plt.figure()
    plt.plot([x[0] for x in data], [y[1] for y in data],'ko') # data

    delta = 0.025
    k = len(means)
    x = np.arange(-2.0, 7.0, delta)
    y = np.arange(-2.0, 7.0, delta)
    X, Y = np.meshgrid(x, y)
    col = ['green', 'red', 'indigo']
    for i in range(k):
        mean = means[i]
        cov = covs[i]
        sigmax = np.sqrt(cov[0][0])
        sigmay = np.sqrt(cov[1][1])
        sigmaxy = cov[0][1]/(sigmax*sigmay)
        Z = mlab.bivariate_normal(X, Y, sigmax, sigmay, mean[0], mean[1], sigmaxy)
        plt.contour(X, Y, Z, colors = col[i])
        plt.title(title)
    plt.rcParams.update({'font.size':16})
    plt.tight_layout()

plot_contours(data, initial_means, initial_covs, 'Initial clusters')

results = EM(data, init_means, init_covs, init_wts, iter_max=12, thresh=1e-4)

plt.show()

