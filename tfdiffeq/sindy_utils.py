from sklearn.linear_model import Lasso
from pysindy.optimizers.stlsq import STLSQ


class STRRidge(STLSQ):
    """
    Sequentially thresholded regression algorithm with Lasso.

    Attempts to minimize the objective function
    :math:`\\|y - Xw\\|^2_2 + alpha \\times \\|w\\|_1`
    by iteratively performing least squares and masking out
    elements of the weight that are below a given threshold.

    Parameters
    ----------
    threshold : float, optional (default 0.1)
        Minimum magnitude for a coefficient in the weight vector.
        Coefficients with magnitude below the threshold are set
        to zero.

    alpha : float, optional (default 0.1)
        Optional L2 (ridge) regularization on the weight vector.

    max_iter : int, optional (default 20)
        Maximum iterations of the optimization algorithm.

    lasso_kw : dict, optional
        Optional keyword arguments to pass to the lasso regression.

    fit_intercept : boolean, optional (default False)
        Whether to calculate the intercept for this model. If set to false, no
        intercept will be used in calculations.

    normalize : boolean, optional (default False)
        This parameter is ignored when fit_intercept is set to False. If True,
        the regressors X will be normalized before regression by subtracting
        the mean and dividing by the l2-norm.

    copy_X : boolean, optional (default True)
        If True, X will be copied; else, it may be overwritten.

    Attributes
    ----------
    coef_ : array, shape (n_features,) or (n_targets, n_features)
        Weight vector(s)

    ind_ : array, shape (n_features,) or (n_targets, n_features)
        Array of 0s and 1s indicating which coefficients of the
        weight vector have not been masked out.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.integrate import odeint
    >>> from pysindy import SINDy
    >>> from pysindy.optimizers import STLSQ
    >>> lorenz = lambda z,t : [10*(z[1] - z[0]),
    >>>                        z[0]*(28 - z[2]) - z[1],
    >>>                        z[0]*z[1] - 8/3*z[2]]
    >>> t = np.arange(0,2,.002)
    >>> x = odeint(lorenz, [-8,8,27], t)
    >>> opt = SRTRidge(threshold=.1, alpha=.5)
    >>> model = SINDy(optimizer=opt)
    >>> model.fit(x, t=t[1]-t[0])
    >>> model.print()
    """

    def __init__(
        self,
        threshold=0.1,
        alpha=0.1,
        max_iter=20,
        lasso_kw=None,
        normalize=False,
        fit_intercept=False,
        copy_X=True,
    ):
        super(STLSQ, self).__init__(
            max_iter=max_iter,
            normalize=normalize,
            fit_intercept=fit_intercept,
            copy_X=copy_X,
        )

        if threshold < 0:
            raise ValueError("threshold cannot be negative")
        if alpha < 0:
            raise ValueError("alpha cannot be negative")

        self.threshold = threshold
        self.alpha = alpha
        self.lasso_kw = lasso_kw

    def _regress(self, x, y):
        """Perform the ridge regression
        """
        kw = self.lasso_kw or {}
        model = Lasso(alpha=self.alpha, fit_intercept=self.fit_intercept,
                      normalize=self.normalize, copy_X=self.copy_X,)
        model.fit(x, y, **kw)
        coef = model.coef_
        self.iters += 1
        return coef
