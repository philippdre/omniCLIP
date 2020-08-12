
"""
    omniCLIP is a CLIP-Seq peak caller

    Copyright (C) 2017 Philipp Boss

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    The code in this file has been adapted from the statsmodels package:
    https://github.com/statsmodels/statsmodels/blob/master/statsmodels/genmod/generalized_linear_model.py
    Thus, for this file the licence of the original file applies additionally.
"""

import pickle
import numpy as np
import scipy
from scipy.sparse import csc_matrix, linalg as sla
import statsmodels
from statsmodels.genmod import families
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.sm_exceptions import PerfectSeparationError
import time

__all__ = ['GLM']


def _check_convergence(criterion, iteration, tol):
    return not (np.fabs(criterion[iteration] - criterion[iteration-1]) > tol)


class sparse_glm(statsmodels.genmod.generalized_linear_model.GLM):
    def __init__(self, endog, exog, family=None, offset=None, exposure=None,
                 missing='none', data_weights=None, **kwargs):

        if hasattr(family, 'safe_links'):
            if (family is not None) and not isinstance(family.link, tuple(family.safe_links)):
                import warnings
                warnings.warn("The %s link function does not respect the domain of the %s family." %
                              (family.link.__class__.__name__, family.__class__.__name__))
        self.endog = endog
        self.exog = exog
        self.offset = offset
        self.exposure = exposure
        if data_weights is not None:
            self.data_weights = data_weights
        if exposure is not None:
            exposure = np.log(exposure)
        if offset is not None:  # this should probably be done upstream
            offset = np.asarray(offset)

        self._check_inputs(family, self.offset, self.exposure, self.endog)
        if offset is None:
            delattr(self, 'offset')
        if exposure is None:
            delattr(self, 'exposure')

    def fit(self, start_params=None, maxiter=100, method='IRLS', tol=1e-8,
            scale=None, cov_type='nonrobust', cov_kwds=None, use_t=None,
            full_output=True, disp=False, max_start_irls=3, data_weights=None,
            **kwargs):
        """Fits a generalized linear model for a given family.

        parameters
        ----------
        start_params : array-like, optional
            Initial guess of the solution for the loglikelihood maximization.
            The default is family-specific and is given by the
            ``family.starting_mu(endog)``. If start_params is given then the
            initial mean will be calculated as ``np.dot(exog, start_params)``.
        maxiter : int, optional
            Default is 100.
        method : string
            Default is 'IRLS' for iteratively reweighted least squares.
            Otherwise gradient optimization is used.
        tol : float
            Convergence tolerance.  Default is 1e-8.
        scale : string or float, optional
            `scale` can be 'X2', 'dev', or a float
            The default value is None, which uses `X2` for Gamma, Gaussian,
            and Inverse Gaussian.
            `X2` is Pearson's chi-square divided by `df_resid`.
            The default is 1 for the Binomial and Poisson families.
            `dev` is the deviance divided by df_resid
        cov_type : string
            The type of parameter estimate covariance matrix to compute.
        cov_kwds : dict-like
            Extra arguments for calculating the covariance of the parameter
            estimates.
        use_t : bool
            If True, the Student t-distribution is used for inference.
        full_output : bool, optional
            Set to True to have all available output in the Results object's
            mle_retvals attribute. The output is dependent on the solver.
            See LikelihoodModelResults notes section for more information.
            Not used if methhod is IRLS.
        disp : bool, optional
            Set to True to print  convergence messages.  Not used if method is
            IRLS.
        max_start_irls : int
            The number of IRLS iterations used to obtain starting
            values for gradient optimization.  Only relevant if
            `method` is set to something other than 'IRLS'.

        Notes
        -----
        This method does not take any extra undocumented ``kwargs``.
        """
        endog = self.endog
        self.df_resid = self.endog.shape[0] - self.exog.shape[1]

        if not isinstance(data_weights, np.ndarray):
            if endog.ndim > 1 and endog.shape[1] == 2:
                data_weights = endog.sum(1)  # weights are total trials
            else:
                data_weights = np.ones((endog.shape[0], 1))

        self.data_weights = data_weights
        if np.shape(self.data_weights) == () and self.data_weights > 1:
            self.data_weights = self.data_weights * np.ones((endog.shape[0]))
        self.scaletype = scale
        if isinstance(self.family, families.Binomial):
            # This checks what kind of data is given for Binomial.
            # Family will need a reference to endog if this is to be removed
            # from preprocessing
            self.endog = self.family.initialize(self.endog)

        # Construct a combined offset/exposure term.  Note that
        # exposure has already been logged if present.
        offset_exposure = 0.
        if hasattr(self, 'offset'):
            offset_exposure = self.offset
        if hasattr(self, 'exposure'):
            offset_exposure = offset_exposure + self.exposure
        self._offset_exposure = offset_exposure

        if method.lower() == "irls_sparse":
            return self._fit_irls_sparse(
                start_params=start_params, maxiter=maxiter,
                tol=tol, scale=scale, cov_type=cov_type,
                cov_kwds=cov_kwds, use_t=use_t, **kwargs)
        else:
            return self._fit_gradient(
                start_params=start_params,
                method=method,
                maxiter=maxiter,
                tol=tol, scale=scale,
                full_output=full_output,
                disp=disp, cov_type=cov_type,
                cov_kwds=cov_kwds, use_t=use_t,
                max_start_irls=max_start_irls,
                **kwargs)

    def _fit_irls_sparse(
            self, start_params=None, maxiter=50, tol=1e-3, scale=None,
            cov_type='nonrobust', cov_kwds=None, use_t=None, **kwargs):
        """Fit a GLM using IRLS.

        Fit a generalized linear mode (GLM) for a given family using
        iteratively reweighted least squares (IRLS).
        """
        if not scipy.sparse.issparse(self.exog):
            raise ValueError("Matrix not sparse")

        endog = self.endog
        wlsexog = self.exog

        if start_params is None:
            mu = self.family.starting_mu(self.endog)
            lin_pred = self.family.predict(mu)
        else:
            # This is a hack for a faster warm start
            start_params[start_params > 1e2] = 1e2
            start_params[start_params < -1e2] = -1e2
            lin_pred = wlsexog.dot(start_params) + self._offset_exposure
            mu = self.family.fitted(lin_pred)

        dev = self.family.deviance(self.endog[self.endog[:, 0] > 0, :], mu[self.endog[:, 0] > 0, :])
        if np.isnan(dev):
            if not (start_params is None):
                # This is a hack for a faster warm start
                start_params[start_params > 1e1] = 1e1
                start_params[start_params < -1e1] = -1e1
                lin_pred = wlsexog.dot(start_params) + self._offset_exposure
                mu = self.family.fitted(lin_pred)

                dev = self.family.deviance(self.endog, mu)

                if np.isnan(dev):
                    pickle.dump([lin_pred, mu, endog, exog, start_params], open('/home/pdrewe/tmp/tmpdump.' + time.asctime().replace(' ', '_') + '.dat', 'w'))
                    raise ValueError("The first guess on the deviance function "
                                 "returned a nan.  This could be a boundary "
                                 " problem and should be reported.")
            else:
                raise ValueError("The first guess on the deviance function "
                             "returned a nan.  This could be a boundary "
                             " problem and should be reported.")

        # first guess on the deviance is assumed to be scaled by 1.
        # params are none to start, so they line up with the deviance
        history = dict(params=[None, start_params], deviance=[np.inf, dev])
        converged = False
        criterion = history['deviance']
        # This special case is used to get the likelihood for a specific
        # params vector.

        if maxiter == 0:
            pass
        for iteration in range(maxiter):
            self.weights = self.data_weights*self.family.weights(mu)
            wlsendog = (lin_pred + self.family.link.deriv(mu) * (self.endog-mu)
                        - self._offset_exposure)
            W = scipy.sparse.diags(self.weights[:, 0], 0)

            # Compute x for current interation
            temp_mat = wlsexog.transpose().dot(W)
            lu = sla.splu(csc_matrix(temp_mat.dot(wlsexog)))
            wls_results = lu.solve(temp_mat.dot(wlsendog))
            wls_results[wls_results > 1e2] = 1e2
            wls_results[wls_results < -1e2] = -1e2

            lin_pred = self.exog.dot(wls_results) + self._offset_exposure
            mu = self.family.fitted(lin_pred)
            history['mu'] = mu
            history['params'].append(wls_results)
            temp_endog = self.endog[:]
            temp_endog[temp_endog < 0] = 0
            history['deviance'].append(self.family.deviance(self.endog, mu))

            if endog.squeeze().ndim == 1 and np.allclose(mu - endog, 0):
                msg = "Perfect separation detected, results not available"
                raise PerfectSeparationError(msg)
            converged = _check_convergence(criterion, iteration, tol)
            if converged:
                break
        self.mu = mu

        history['iteration'] = iteration + 1

        return [wls_results, history]

    def _check_inputs(self, family, offset, exposure, endog):
        # Default family is Gaussian
        if family is None:
            family = families.Gaussian()
        self.family = family

        if exposure is not None:
            if not isinstance(self.family.link, families.links.Log):
                raise ValueError("exposure can only be used with the log "
                                 "link function")
            elif exposure.shape[0] != endog.shape[0]:
                raise ValueError("exposure is not the same length as endog")

        if offset is not None:
            if offset.shape[0] != endog.shape[0]:
                raise ValueError("offset is not the same length as endog")
