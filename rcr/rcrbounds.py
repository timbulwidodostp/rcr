"""
Calculate RCR bounds for linear causal effects (Krauth 2016)

Author:       Brian Krauth
              Department of Economics
              Simon Fraser University
Usage:

This Python package is meant to be used in one of two ways:

1. As the Python module rcrbounds.  To fit an RCR model in
   Python:

    1. Use the RCR function to set up the model as an
       RCR object.
    2. Use the RCR object's fit() method to fit the model,
       and return an RCRResults object.
    3. Use the RCRResults objects summary() method to
       show a summary table of results.

    For additional details and options, see the associated
    docstrings:
        rcrbounds.RCR?
        rcrbounds.RCR.fit?
        rcrbounds.RCRResults?
        rcrbounds.RCRResults.summary?

2. As a Python script to be called from the Stata rcr command.
   This call is invisible to most Stata users, but if you are
   debugging or trying to understand the code, view the
   docstring for the stata_exe() function.
"""
# pylint: disable=too-many-lines
# Standard library imports
import sys
import warnings
from datetime import datetime

# Third party imports
import numpy as np
from numpy.linalg import inv
import pandas as pd
import scipy.stats
try:
    import statsmodels.iolib as si
    import statsmodels.iolib.summary as su
    import matplotlib.pyplot as plt
except ImportError:
    pass

# Local application imports
# (none)

# Global variables

LOGFILE = None

# Class definitions


class RCR:
    """
    A class to represent a regression model for RCR analysis.

    Parameters
    ----------
    endog : array_like
        A nrows x 2 array.  first column represents the outcome
        (dependent variable) and the second column represents the
        treatment (explanatory variable of interest).
    exog : array_like
        A nrows x k array of control variables. The first column should
        be an intercept.  See :func:`statsmodels.tools.add_constant`
        to add an intercept.
    weights : array_like or None
        An optional nrows x 1 array of weights.
        Default is None.
    groupvar : array_like or None
        An optional nrows x 1 array of group ID variables for the
        calculation of cluster-robust standard errors.
        Default is None.
    rc_range: array_like
        An optional 1-d array of the form [rcL, rcH]
        where rcL is the lower bound and rcH is the
        upper bound for the RCR parameter rc.  rcL can
        be -inf to indicate no lower bound, and rcH can
        be inf to indicate no upper bound.  rcL should always
        be <= rcH.
        Default is [0.0, 1.0].
    cov_type : str
        The method used to estimate the covariance matrix
        of parameter estimates. Currently-available options
        are 'nonrobust' (the usual standard errors for
        a random sample) and 'cluster' (cluster-robust
        standard errors)
        Default is 'nonrobust'.
    vceadj : float
        Optional degrees of freedom adjustment factor. The covariance
        matrix of parameters will be multiplied by vceadj.
        Default is no adjustment (vceadj = 1.0).
    citype: str
        Optional confidence interval type for the causal effect
        of interest.  Options include "conservative" (ignores
        the width of the identified set), "Imbens-Manski"
        (accounts for the width of the identified set),
        "upper" (one-tailed upper CI), and "lower" (one-tailed
        lower CI).
        Default is "conservative"
    cilevel : float
        Optional confidence level on a 0 to 100 scale for confidence
        interval calculations.
        Default is 95.

    Attributes
    ----------
    endog : ndarray
        the array of endogenous variables.
    exog : ndarray
        the array of exogenous variables
    rc_range : ndarray
        the array of rc values.
    weights : ndarray or None
        the array of weights.
    groupvar : ndarray or None
        the array of group ID variables
    cov_type : str
        the method used to estimate the covariance matrix
    vceadj : float
        the degrees of freedom adjustment factor
    citype: str
        the confidence interval type
    cilevel : float
        the confidence level
    endog_names, exog_names : ndarray of str
        the names of the variables in endog and exog.
    depvar, treatvar, controlvars : str
        the names of the dependent, treatment and
        control variables (from endog_names and
        exog_names)
    nobs : float
        the number of observations.

    Methods
    ------------
    fit()
        Estimate the RCR model.
    copy()
        Create a copy of the RCR model object, with
        modificatios as specified by any keyword arguments
        provided.
    rcvals()
        Calculate the rc(effect) function associated
        with the RCR model.  This is used mostly for
        plots.

    See also
    --------
    RCRResults

    Notes
    -----
    The RCR class is patterned after
    statsmodels.regression.linear_model.OLS.

    Examples
    --------
    Setup:
    >>> dat = pd.read_stata("http://www.sfu.ca/~bkrauth/code/rcr_example.dta")
    >>> dat["Intercept"] = 1.0
    >>> endog = dat[["SAT", "Small_Class"]]
    >>> exog = dat[["Intercept", "White_Asian", "Girl",
    ...             "Free_Lunch", "White_Teacher", "Teacher_Experience",
    ...             "Masters_Degree"]]

    Create RCR model object:
    >>> model = RCR(endog, exog)

    Report attributes of the model
    >>> print(model.nobs)
    5839

    Use the fit() method to fit the model
    >>> results = model.fit()

    Report results using attributes of the RCRResults object
    >>> print(results.params)
    [12.31059909  8.16970997 28.93548917  5.13504376  5.20150257]
    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self,
                 endog,
                 exog,
                 rc_range=np.array([0.0, 1.0]),
                 cov_type="nonrobust",
                 vceadj=1.0,
                 citype="conservative",
                 cilevel=95,
                 weights=None,
                 groupvar=None):
        """Constructs the RCR object."""
        # pylint: disable=too-many-arguments
        self.endog = np.asarray(endog)
        check_endog(self.endog)
        nrows = self.endog.shape[0]
        self.exog = np.asarray(exog)
        check_exog(self.exog, nrows)
        if weights is None:
            self.weights = None
            self.nobs = nrows
        else:
            self.weights = np.asarray(weights)
            self.weights_name = get_column_names(weights,
                                                 default_names="(no name)")
            check_weights(self.weights, nrows)
            self.nobs = sum(weights > 0)
        if groupvar is not None:
            self.groupvar = np.asarray(groupvar)
            self.groupvar_name = get_column_names(groupvar,
                                                  default_names="(no name)")
            if weights is None:
                grp = pd.Series(np.ones(nrows)).groupby(groupvar).sum() > 0
                self.ngroups = sum(grp)
            else:
                grp = pd.Series(weights).groupby(groupvar).sum() > 0
                self.ngroups = sum(grp)
        else:
            self.groupvar = None
        endog_names_default = ["y", "treatment"]
        self.endog_names = get_column_names(endog,
                                            default_names=endog_names_default)
        exog_names_default = ["x" + str(x) for
                              x in
                              list(range(1, exog.shape[1]))]
        exog_names_default = ["Intercept"] + exog_names_default
        self.exog_names = get_column_names(exog,
                                           default_names=exog_names_default)
        self.depvar = self.endog_names[0]
        self.treatvar = self.endog_names[1]
        self.controlvars = " ".join([str(item) for
                                     item in
                                     self.exog_names[1:]])
        self.rc_range = np.asarray(rc_range)
        self.cov_type = cov_type
        self.vceadj = vceadj
        self.citype = citype
        self.cilevel = cilevel
        check_rc(self.rc_range)
        check_covinfo(cov_type, vceadj)
        check_ci(cilevel, citype)

    def copy(self, **kwargs):
        """Copies (and possibly modifies) an RCR object."""
        endog = kwargs.get("endog")
        exog = kwargs.get("exog")
        rc_range = kwargs.get("rc_range")
        cov_type = kwargs.get("cov_type")
        vceadj = kwargs.get("vceadj")
        citype = kwargs.get("citype")
        cilevel = kwargs.get("cilevel")
        weights = kwargs.get("weights")
        groupvar = kwargs.get("groupvar")
        if endog is None:
            endog = pd.DataFrame(self.endog,
                                 columns=self.endog_names)
        if exog is None:
            exog = pd.DataFrame(self.exog,
                                columns=self.exog_names)
        if rc_range is None:
            rc_range = self.rc_range
        if cov_type is None:
            cov_type = self.cov_type
        if vceadj is None:
            vceadj = self.vceadj
        if citype is None:
            citype = self.citype
        if cilevel is None:
            cilevel = self.cilevel
        if weights is None and self.weights is not None:
            weights = pd.DataFrame(self.weights,
                                   columns=self.weights_name)
        if groupvar is None and self.groupvar is not None:
            groupvar = pd.DataFrame(self.groupvar,
                                    columns=self.groupvar_name)
        return RCR(endog=endog,
                   exog=exog,
                   rc_range=rc_range,
                   cov_type=cov_type,
                   vceadj=vceadj,
                   citype=citype,
                   cilevel=cilevel,
                   weights=weights,
                   groupvar=groupvar)

    def _get_mv(self,
                estimate_cov=False):
        """Calculates the moment vector used in RCR estimation."""
        xyz = np.concatenate((self.exog, self.endog), axis=1)
        k = xyz.shape[1]
        msk = np.triu(np.full((k, k), True)).flatten()
        xyzzyx = np.apply_along_axis(bkouter, 1, xyz, msk=msk)[:, 1:]
        moment_vector = np.average(xyzzyx, axis=0, weights=self.weights)
        if estimate_cov:
            if self.weights is None and self.cov_type != "cluster":
                fac = 1/self.nobs
                cov_mv = fac*np.cov(xyzzyx,
                                    rowvar=False)
            elif self.weights is not None and self.cov_type != "cluster":
                fac = sum((self.weights/sum(self.weights)) ** 2)
                cov_mv = fac*np.cov(xyzzyx,
                                    rowvar=False,
                                    aweights=self.weights)
            else:
                cov_mv = robust_cov(xyzzyx,
                                    groupvar=self.groupvar,
                                    weights=self.weights)
            return moment_vector, cov_mv
        return moment_vector

    def rcvals(self,
               effectvals=np.linspace(-50, 50, 100),
               add_effectinf=False):
        """Estimates rc() for a set of values."""
        effectvals = np.asarray(effectvals).flatten()
        moment_vector = self._get_mv()
        simplified_moments = simplify_moments(moment_vector)
        effect_inf = effectinf(moment_vector)
        rcvals = rcfast(effectvals, simplified_moments)
        if add_effectinf and min(effectvals) <= effect_inf <= max(effectvals):
            effectvals = np.append(effectvals, [effect_inf])
            rcvals = np.append(rcvals, [np.nan])
            msk = np.argsort(effectvals)
            rcvals = rcvals[msk]
            effectvals = effectvals[msk]
        return rcvals, effectvals

    def fit(self,
            **kwargs):
        """
        Estimates an RCR model.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments that will override any options
            specified when constructing the model.

        Returns
        -------
        RCRResults
            the model estimation results.

        See Also
        --------
        RCRResults
            the results container.
        """
        # pylint: disable=protected-access
        if not kwargs:
            model = self
        else:
            model = self.copy(**kwargs)
        moment_vector, cov_mv = model._get_mv(estimate_cov=True)
        (result_matrix, effectvec, rcvec) = \
            estimate_model(moment_vector, model.rc_range)
        params = result_matrix[:, 0]
        cov_params = (model.vceadj *
                      result_matrix[:, 1:] @
                      cov_mv @
                      result_matrix[:, 1:].T)
        details = np.array([effectvec, rcvec])
        return RCRResults(model=model,
                          params=params,
                          cov_params=cov_params,
                          details=details)


class RCRResults:
    """
    Results class for an RCR model.

    Parameters
    ----------
    model : RCR object
        The RCR object representing the model to be estimated.
        See rcrbounds.RCR for details.
    params : ndarray
        A 5-element ndarray representing estimates for the point-identified
        parameters [rcInf, effectInf, rc0, effectL, effectH]
    cov_params : ndarray
        A 5 x 5 ndarray representing the estimated covariance matrix
        for the esimates in params.

    Attributes
    ----------
    model : RCR object
        the model that has been estimated.
    params : ndarray
        the estimated parameters.
    cov_params : ndarray
        the covariance matrix for the parameter estimates.
    param_names : list
        the parameter names.
    details : ndarray
        a d x 2 array representing the rc(effect) function
        the first column is a set of effect values,
        the second column is the estimated rc(effect)
        for that value.
    nobs : float
        the number of observations.

    Methods
    ------------
    params_se()
        standard errors for params.
    params_z()
        z-statistics for params.
    params_pvalue()
        asymptotic p-values for params.
    params_ci()
        confidence intervals for params.
    effect_ci()
        confidence interval for the causal effect.
    test_effect()
        hypothesis test for the causal effect.
    summary()
        summary of results.
    rcrplot()
        plot of results.

    See also
    --------
    RCR class.

    Notes
    -----
    The RCRResults class is patterned after
    statsmodels.regression.linear_model.RegressionResults.

    Examples
    --------
    Setup:
    >>> dat = pd.read_stata("http://www.sfu.ca/~bkrauth/code/rcr_example.dta")
    >>> dat["Intercept"] = 1.0
    >>> endog = dat[["SAT", "Small_Class"]]
    >>> exog = dat[["Intercept", "White_Asian", "Girl",
    ...             "Free_Lunch", "White_Teacher", "Teacher_Experience",
    ...             "Masters_Degree"]]
    >>> model = RCR(endog, exog)
    >>> results = model.fit()

    Report results using attributes of the RCRResults object
    >>> print(results.params)
    [12.31059909  8.16970997 28.93548917  5.13504376  5.20150257]

    Report results using methods of the RCRResults object
    >>> print(results.params_se())
    [  2.09826858  30.60745128 108.51947421   0.95693751   0.6564318 ]
    >>> result_summary = results.summary()

    """
    # pylint: disable=too-many-instance-attributes
    def __init__(self,
                 model,
                 params,
                 cov_params,
                 details):
        """Constructs the RCRResults object."""
        # pylint: disable=too-many-arguments
        self.model = model
        self.params = params
        self.param_names = ["rcInf",
                            "effectInf",
                            "rc0",
                            "effectL",
                            "effectH"]
        self.cov_params = cov_params
        self.details = details

    def params_se(self):
        """Calcuates standard errors for RCR parameter estimates."""
        return np.sqrt(np.diag(self.cov_params))

    def params_z(self):
        """Calcuates z-statistics for RCR parameter estimates."""
        return self.params / self.params_se()

    def params_pvalue(self):
        """Calcuates asymptotic p-values for RCR parameter estimates."""
        alpha = scipy.stats.norm.cdf(np.abs(self.params / self.params_se()))
        return 2 * (1.0 - alpha)

    def params_ci(self, cilevel=None):
        """
        Calcuates asymptotic confidence intervals for RCR parameters.

        Parameters
        ----------
        cilevel : float
            Optional confidence level on a 0 to 100 scale for confidence
            interval calculations.
            Default is the value set in the RCR model object.

        Returns
        -------
        a 2 x 5 ndarrray representing the confidence intervals
        """
        if cilevel is None:
            cilevel = self.model.cilevel
        check_ci(cilevel)
        crit = scipy.stats.norm.ppf((100 + cilevel) / 200)
        return np.array([self.params - crit * self.params_se(),
                         self.params + crit * self.params_se()])

    def effect_ci(self,
                  cilevel=None,
                  citype="conservative"):
        """
        Calculates asymptotic confidence intervals for the RCR causal effect.

        Parameters
        ----------
        cilevel : float
            Optional confidence level on a 0 to 100 scale for confidence
            interval calculations.
            Default is the value set in the RCR model object.
        citype: str
            Optional confidence interval type for the causal effect
            of interest.  Options include "conservative" (ignores
            the width of the identified set), "Imbens-Manski"
            (accounts for the width of the identified set),
            "upper" (one-tailed upper CI), and "lower" (one-tailed
            lower CI).
            Default is the value set in the RCR model object.

        Returns
        -------
        a length-2 ndarrray representing the confidence interval
        """
        if citype == "conservative":
            effect_ci = self.effect_ci_conservative(cilevel=cilevel)
        elif citype == "upper":
            effect_ci = self.effect_ci_upper(cilevel=cilevel)
        elif citype == "lower":
            effect_ci = self.effect_ci_lower(cilevel=cilevel)
        elif citype == "Imbens-Manski":
            effect_ci = self.effect_ci_imbensmanski(cilevel=cilevel)
        else:
            effect_ci = np.array([np.nan, np.nan])
        return effect_ci

    def effect_ci_conservative(self, cilevel=None):
        """Calcuates conservative confidence interval for causal effect."""
        if cilevel is None:
            cilevel = self.model.cilevel
        crit = scipy.stats.norm.ppf((100 + cilevel) / 200)
        ci_lb = self.params[3] - crit * self.params_se()[3]
        ci_ub = self.params[4] + crit * self.params_se()[4]
        return np.array([ci_lb, ci_ub])

    def effect_ci_upper(self, cilevel=None):
        """Calcuates upper confidence interval for causal effect."""
        if cilevel is None:
            cilevel = self.model.cilevel
        crit = scipy.stats.norm.ppf(cilevel / 100)
        ci_lb = self.params[3] - crit * self.params_se()[3]
        ci_ub = np.inf
        return np.array([ci_lb, ci_ub])

    def effect_ci_lower(self, cilevel=None):
        """Calcuates lower confidence interval for causal effect."""
        if cilevel is None:
            cilevel = self.model.cilevel
        crit = scipy.stats.norm.ppf(cilevel / 100)
        ci_lb = -np.inf
        ci_ub = self.params[4] + crit * self.params_se()[4]
        return np.array([ci_lb, ci_ub])

    def effect_ci_imbensmanski(self, cilevel=None):
        """Calcuates Imbens-Manski confidence interval for causal effect."""
        if cilevel is None:
            cilevel = self.model.cilevel
        cv_min = scipy.stats.norm.ppf(1 - ((100 - cilevel) / 100.0))
        cv_mid = cv_min
        cv_max = scipy.stats.norm.ppf(1 - ((100 - cilevel) / 200.0))
        params_se = self.params_se()
        delta = ((self.params[4] - self.params[3]) /
                 max(params_se[3], params_se[4]))
        if np.isfinite(delta):
            while (cv_max - cv_min) > 0.000001:
                cv_mid = (cv_min + cv_max) / 2.0
                if (scipy.stats.norm.cdf(cv_mid + delta) -
                   scipy.stats.norm.cdf(-cv_mid)) < (cilevel / 100):
                    cv_min = cv_mid
                else:
                    cv_max = cv_mid
        if params_se[3] > 0:
            ci_lb = self.params[3]-(cv_mid * params_se[3])
        else:
            ci_lb = -np.inf
        if params_se[4] > 0:
            ci_ub = self.params[4]+(cv_mid * params_se[4])
        else:
            ci_ub = np.inf
        return np.array([ci_lb, ci_ub])

    def test_effect(self, h0_value=0.0):
        """
        Conducts a hypothesis test for the RCR causal effect.

        Parameters
        ----------
        h0_value : float
            Optional value for causal effect under the null hypothesis.
            Default is zero.

        Returns
        -------
        the p-value for the test of the null hypothesis
            H0: effect = h0_value

        Notes
        -------
        This test works by inverting the Imbens-Manski confidence interval.
        That is, the function reports a p-value defined as (1 - L/100)
        where L is the highest confidence level at which h0 is outside
        of the L% confidence interval.  For example, the p-value will
        be less than 0.05 (reject the null at 5%) if h0 is outside of
        the 95% confidence interval.  Since the test works by inverting
        the confidence interval, there is no associated test statistic
        to report.

        See also
        --------
        RCRResults.effect_ci()
        """
        low = 0.0
        high = 100.0
        mid = 50.0
        if self.params[3] <= h0_value <= self.params[4]:
            pvalue = 1.0
        else:
            while (high - low) > 0.00001:
                mid = (high + low) / 2.0
                current_ci = self.effect_ci_imbensmanski(cilevel=mid)
                if current_ci[0] <= h0_value <= current_ci[1]:
                    high = mid
                else:
                    low = mid
            pvalue = 1.0 - low/100.0
        return pvalue

    def rcrplot(self,
                ax=None,
                xlim=(-50, 50),
                ylim=None,
                tsline=False,
                lsline=False,
                idset=False,
                title=None,
                xlabel=r"Effect ($\beta_x$)",
                ylabel=r"Relative correlation ($\lambda$)",
                flabel=r"$\lambda(\beta_x)$ function",
                tslabel=r"$\beta_x^{\infty}$",
                lslabel=r"$\lambda^{\infty}$",
                idlabels=(r"RC bounds $[\lambda^L,\lambda^H]$",
                          r"Identified set $[\beta_x^L,\beta_x^H]$"),
                tss="--",
                lss="-.",
                fcolor="C0",
                tscolor="0.75",
                lscolor="0.75",
                idcolors=("C0", "C0"),
                idalphas=(0.25, 0.75),
                legend=False):
        """
        Creates a plot of RCR estimation results.

        Parameters
        ----------
        ax : a matplotlib axes object`or None
            Axis object to return/modify.
            Default is None.
        xlim : array-like
            Range of values for x-axis
            Default is (-50, 50),
        ylim : array-like or None
            Range of values for y-axis.  If None,
            the y-axis will adjust to fit the data.
            Default is None
        tsline : bool
            Optional flag to show a line for effect_inf
            Default is False,
        lsline : bool
            Optional flag to show a line for rc_inf
            Default is False,
        idset : bool
            Optional flag to show the identified set
            Default is False,
        title : str or None
            Optional plot title.
            Default is None,
        xlabel : str
            Optional label for x axis.
            Default is r"Effect ($\beta_x$)",
        ylabel : str
            Optional label for y axis.
        flabel : str
            Optional label for rc(effect) function.
        tslabel : str
            Optional label for effect_inf line.
        lslabel : str
            Optional label for rc_inf line.
        idlabels : (str, str)
            Optional labels for identified set.
        tss : str
            Optional line type for effect_inf
        lss : str
            Optional line type for rc_inf
        fcolor : str
            Optional color specification for rc(effect) function
            Default is "C0",
        tscolor : str
            Optional color specification for effect_inf line
            Default is "0.75",
        lscolor : str
            Optional color specification for rc_inf line
            Default is "0.75",
        idcolors : (str, str)
            Optional color specifications for identified set
            Default is ("C0", "C0"),
        idalphas : (str, str)
            Optional alpha specifications for identified set
            Default is=(0.25, 0.75),
        legend : bool
            Optional flag to include legend
            Default is=False

        Returns
        -------
        ax
            a matplotlib axes object

        Notes
        -------

        See also
        --------

        """
        # pylint: disable=too-many-arguments,too-many-locals,invalid-name
        xlim = np.sort(np.asarray(xlim))
        if len(xlim) == 2:
            xgrid = np.linspace(xlim[0], xlim[1], num=100)
        else:
            xgrid = xlim
        rcvals, effectvals = self.model.rcvals(effectvals=xgrid,
                                               add_effectinf=True)
        if ax is None:
            ax = plt.gca()
            ax.clear()
        ax.plot(effectvals,
                rcvals,
                label=flabel,
                color=fcolor)
        if ylim is not None:
            ax.set_ylim(ylim[0], ylim[1])
        if tsline is True:
            effect_inf = self.params[1]
            if xlim[0] <= effect_inf <= xlim[-1]:
                ax.axvline(effect_inf,
                           ls=tss,
                           color=tscolor,
                           label=tslabel)
        if lsline is True:
            rc_inf = self.params[0]
            if min(rcvals) <= rc_inf <= max(rcvals):
                ax.axhline(rc_inf,
                           ls=lss,
                           color=lscolor,
                           label=lslabel)
        if idset is True:
            ax.axhspan(self.model.rc_range[0],
                       self.model.rc_range[1],
                       color=idcolors[0],
                       alpha=idalphas[0],
                       label=idlabels[0])
            ax.axvspan(self.params[3],
                       self.params[4],
                       color=idcolors[1],
                       alpha=idalphas[1],
                       label=idlabels[1])
        if title is not None:
            ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if legend:
            ax.legend()
        return ax

    def summary(self,
                citype=None,
                cilevel=None,
                tableformats=None):
        """
        Displays a summary of RCR results.

        Parameters
        ----------
        cilevel : float
            the confidence level for the confidence intervals, on
            a scale of 0 to 100.  Default is the cilevel
            attribute of the RCRResults object.
        citype : "conservative", "upper", "lower" or "Imbens-Manski"
            the method to be used in calculating the confidence
            interval for the causal effect effect. Default is
            the citype attribute of the RCRResults object.
        tableformats: list
            a list of formatting strings to use for the table
            of parameter estimates. If the length of tableformats
            is <6, elements will be repeated as needed.  Default
            is ["%9.4f", "%9.3f", "%9.3f", "%9.3f", "%9.3f", "%9.3f"].

        See also
        --------
        RCR class, RCRResults class

        Notes
        -----
        The summary() method returns a
        statsmodels.iolib.summary.Summary object.

        """
        # pylint: disable=too-many-locals
        if citype is None:
            citype = self.model.citype
        if cilevel is None:
            cilevel = self.model.cilevel
        if tableformats is None:
            tableformats = ["%9.4f", "%9.3f", "%9.3f",
                            "%9.3f", "%9.3f", "%9.3f"]
        tableformats = (tableformats*6)[0:6]
        outmat = pd.DataFrame(index=self.param_names)
        outmat["b"] = self.params
        outmat["se"] = self.params_se()
        outmat["z"] = self.params_z()
        outmat["pz"] = self.params_pvalue()
        params_ci = self.params_ci(cilevel=cilevel)
        outmat["ciL"] = params_ci[0, :]
        outmat["ciH"] = params_ci[1, :]
        effect_ci = self.effect_ci(cilevel=cilevel, citype=citype)
        ncontrols = self.model.exog.shape[1] - 1
        table1data = [[self.model.depvar,
                       self.model.treatvar],
                      [datetime.now().strftime("%a, %d %b %Y"),
                       self.model.rc_range[0]],
                      [datetime.now().strftime("%H:%M:%S"),
                       self.model.rc_range[1]],
                      [self.model.nobs,
                       ncontrols],
                      [self.model.cov_type,
                       self.model.vceadj]]
        table1stub1 = ["Dep. Variable",
                       "Date",
                       "Time",
                       "No. Observations",
                       "Covariance Type"]
        table1stub2 = ["Treatment Variable",
                       "Lower bound on rc",
                       "Upper bound on rc",
                       "No. Controls",
                       "Cov. adjustment factor"]
        if self.model.cov_type == "cluster":
            table1data.append([self.model.groupvar_name,
                               self.model.ngroups])
            table1stub1.append("Cluster variable:")
            table1stub2.append("No. Clusters")
        if self.model.weights is not None:
            table1data.append([self.model.weights_name,
                               ""])
            table1stub1.append("Weight variable:")
            table1stub2.append("")
        table1 = si.table.SimpleTable(table1data,
                                      stubs=table1stub1,
                                      title="RCR Regression Results")
        table1.insert_stubs(2, table1stub2)
        table2data = np.asarray(outmat)
        table2headers = ["coef",
                         "std err",
                         "z",
                         "P>|z|",
                         "[" + str((100 - cilevel)/200),
                         str((100 + cilevel)/200) + "]"]
        table2stubs = self.param_names
        table2 = si.table.SimpleTable(table2data,
                                      headers=table2headers,
                                      stubs=table2stubs,
                                      data_fmts=tableformats)
        table3data = [[effect_ci[0], effect_ci[1]]]
        table3stubs = ["effect_ci (" +
                       citype +
                       ")                            "]
        table3 = si.table.SimpleTable(table3data,
                                      stubs=table3stubs,
                                      data_fmts=tableformats[5:])
        obj = su.Summary()
        obj.tables = [table1, table2, table3]
        cstr = f"Control Variables: {self.model.controlvars}"
        obj.add_extra_txt([cstr])
        return obj


# Model calculation functions


def estimate_model(moment_vector, rc_range):
    """Estimates the RCR model.

    Parameters
    ----------
    moment_vector : ndarray of floats
        its elements will be interpreted as the upper triangle of the
        (estimated) second moment matrix E(W'W), where W = [1 X Y Z].
        It is normally constructed by Stata.
    rc_range : ndarray of floats
        its elements rc values to consider

    Returns
    -------
    result_matrix : ndarray
        an array of parameter estimates and gradients
    theetavec, rcvec : ndarray
        the rc(effect) function, estimated at a large
        number of points

    """
    write_to_logfile("Estimating model.\n")
    result_matrix = np.full((len(rc_range) + 3,
                             len(moment_vector) + 1),
                            float('nan'))
    # Check to make sure the moments are consistent
    valid, identified = check_moments(moment_vector)
    # If moments are invalid, just stop there
    if not valid:
        return result_matrix
    # If model is not identified, just stop there
    if not identified:
        return result_matrix
    # We have closed forms for the global parameters rc_inf, effect_inf,
    # and rc(0), so we just estimate them directly.
    result_matrix[0, ] = estimate_parameter(rcinf, moment_vector)
    result_matrix[1, ] = estimate_parameter(effectinf, moment_vector)
    result_matrix[2, ] = estimate_parameter(rc0_fun, moment_vector)
    # Here we get to the main estimation problem.  We need to find the range
    # of effect values consistent with the rc(effect) function falling in
    # rc_range.  We have a closed form solution for rc(effect), but
    # finding its inverse is an iterative problem.
    #
    # STEP 1: Estimate effect_SEGMENTS, which is a global real vector
    #         indicating all critical points (i.e., points where the
    #         derivative is zero or nonexistent) of the function
    #         rc(effect).  The function is continuous and monotonic
    #         between these points. Note that we don't know a priori how many
    #         critical points there will be, and so we don't know how big
    #         effect_SEGMENTS will be.
    effect_segments, effectvec, rcvec = \
        estimate_effect_segments(moment_vector)
    # STEP 2: For each row of rc_range (i.e., each pair of rc values):
    # do i=1,size(rc_range,1)
    # j is the row in result_matrix corresponding to rc_range(i,:)
    # j = 2+2*i
    #  Estimate the corresponding effect range, and put it in result_matrix
    result_matrix[3:5, :] = estimate_effect(moment_vector,
                                            rc_range,
                                            effect_segments)
    return result_matrix, effectvec, rcvec


def estimate_effect_segments(moment_vector):
    """Constructs segments over which rc(effect) is monotonic."""
    imax = 30000   # A bigger number produces an FP overflow in fortran
    simplified_moments = simplify_moments(moment_vector)
    effect_inf = effectinf(moment_vector)
    # effectMAX is the largest value of effect for which we can calculate both
    # rc(effect) and rc(-effect) without generating a floating point
    # exception.
    effectmax = np.sqrt(sys.float_info.max /
                        max(1.0,
                            simplified_moments[4],
                            simplified_moments[1] - simplified_moments[4]))
    # The calculation above seems clever, but it turns out not to always work.
    # So I've put in a hard limit as well
    effectmax = min(1.0e100, effectmax)
    # Create a infting set of effect values at which to calculate rc
    effectvec = np.sort(np.append(np.linspace(-50.0, 50.0, imax - 2),
                                  (effectmax, -effectmax)))
    if np.isfinite(effect_inf):
        # Figure out where effect_inf lies in effectvec
        i = np.sum(effectvec < effect_inf)
        # If i=0 or i=k, then effect_inf is finite but outside of
        # [-effectmax,effectmax]. This is unlikely, but we should check.
        if 0 < i < imax:
            # Adjust i to ensure that -effectmax and effectmax are still
            # included in effectvec
            i = min(max(i, 2), imax - 2)
            # Replace the two elements of effectvec that bracket effect_inf
            # with two more carefully-chosen numbers.  See BRACKET_EFFECT_INF
            # for details
            bracket = bracket_effect_inf(moment_vector)
            if bracket is not None:
                effectvec[i-1: i+1] = bracket
            # There is a potential bug here.  The bracket_effect_inf
            # function is used to take the two values in effectvec that are
            # closest to effect_inf and replace them with values that are
            # guaranteed to give finite and nonzero rc.  But there's
            # nothing to guarantee that these are still the two values in
            # effectvec that are the closest to effect_inf.
            assert effectvec[i-2] < effectvec[i-1]
            assert effectvec[i] < effectvec[i+1]
    # Re-sort effectvec
    effectvec = np.sort(effectvec)
    # Calculate rc for every effect in effectvec
    rcvec = rcfast(effectvec, simplify_moments(moment_vector))
    # LOCALMIN = True if the corresponding element of effectVEC appears to be
    # a local minimum
    localmin = ((rcvec[1:imax-1] < rcvec[0:imax-2]) &
                (rcvec[1:imax-1] < rcvec[2:imax]))
    # The end points are not local minima
    localmin = np.append(np.insert(localmin, [0], [False]), False)
    # LOCALMAX = True if the corresponding element of effectVEC appears to be
    # a local maximum
    localmax = ((rcvec[1:imax-1] > rcvec[0:imax-2]) &
                (rcvec[1:imax-1] > rcvec[2:imax]))
    # The end points are not local max`ima
    localmax = np.append(np.insert(localmax, [0], [False]), False)
    # Figure out where effect_inf lies in effectVEC.  We need to do this
    # calculation again because we sorted effectVEC
    if np.isfinite(effect_inf):
        i = np.sum(effectvec < effect_inf)
        if 0 < i < imax:
            # The two values bracketing effect_inf are never local optima
            localmin[i-1:i+1] = False
            localmax[i-1:i+1] = False
    # Right now, we only have approximate local optima.  We need to apply
    # an iterative optimization algorithm to improve the precision.
    # do j=1,size(localmin)
    for j in range(1, len(localmin)):
        if localmin[j-1]:
            effectvec[j-1] = brent(effectvec[j-2],
                                   effectvec[j-1],
                                   effectvec[j],
                                   rcfast,
                                   1.0e-10,
                                   simplify_moments(moment_vector))
        elif localmax[j-1]:
            effectvec[j-1] = brent(effectvec[j-2],
                                   effectvec[j-1],
                                   effectvec[j],
                                   negative_rcfast,
                                   1.0e-10,
                                   simplify_moments(moment_vector))
    # Now we are ready to create effect_SEGMENTS.
    if np.isfinite(effect_inf) and 0 < i < imax:
        # effect_SEGMENTS contains the two limits (-Inf,+Inf), the pair of
        # values that bracket effect_inf, and any local optima
        effect_segments = np.append(np.concatenate([effectvec[i-1:i+1],
                                                   effectvec[localmin],
                                                   effectvec[localmax]]),
                                    (-effectmax, effectmax))
    else:
        # If effect_inf is not finite, then we have two less elements in
        # effect_SEGMENTS
        effect_segments = np.concatenate([effectvec[i-1:i+1],
                                         effectvec[localmin],
                                         effectvec[localmax]])
    # Sort the result (definitely necessary)
    effect_segments = np.sort(effect_segments)
    return effect_segments, effectvec, rcvec


def bracket_effect_inf(moment_vector):
    """Finds effect valus close to effect_inf."""
    # Get the value of effect_inf.  If we are in this function it should be
    # finite.
    effect_inf = effectinf(moment_vector)
    # Get the limit of rc(effect) as effect approaches effect_inf,from below
    # and from above. These limits are generally not finite.
    simplified_moments = simplify_moments(moment_vector)
    # If this condition holds, no need to find a bracket (and the code
    # below won't work anyway)
    if simplified_moments[2] == (simplified_moments[5] *
                                 simplified_moments[1] /
                                 simplified_moments[4]):
        return None
    # We may want to use np.inf here
    # NOTE: the np.sign seems extraneous here.
    true_limit = (np.array((1.0, -1.0)) *
                  np.sign(simplified_moments[2] -
                          simplified_moments[5] *
                          simplified_moments[1] /
                          simplified_moments[4]) *
                  sys.float_info.max)
    # Pick a default value
    bracket = None
    j = 0
    for i in range(1, 101):
        # For the candidate bracket, consider effect_inf plus or minus some
        # small number epsilon (epsilon gets smaller each iteration)
        candidate = (effect_inf +
                     np.array((-1.0, 1.0)) * max(abs(effect_inf), 1.0)*0.1**i)
        # To be a good bracket, candidate must satisfy some conditions:
        #    1. The bracket must be wide enough that the system can tell that
        #       CANDIDATE(1) < effect_inf < CANDIDATE(2)
        #    2. The bracket must be narrow enough that rc(candidate) is
        #       the same sign as true_limit.
        #    3. The bracket must be wide enough that rc(candidate) is
        #       finite and nonzero. If candidate is very close to effect_inf,
        #       then the calculated rc(candidate) can be *either* NaN or
        #       zero.  The reason for this is that rc(candidate) is a
        #       ratio of two things that are going to zero.  Approximation
        #       error will eventually make both the numerator and denominator
        #       indistingushable from zero (NaN), but sometimes the numerator
        #       will reach indistinguishable-from-zero faster (giving zero
        #       for the ratio).
        if candidate[0] < effect_inf < candidate[1]:
            tmp2 = rcfast(candidate, simplified_moments)
            if (np.isfinite(tmp2).all() and
               (tmp2[0]*np.sign(true_limit[0]) > 0.0) and
               (tmp2[1]*np.sign(true_limit[1]) > 0.0)):
                j = i
                bracket = candidate
            else:
                continue
    if j == 0:
        msg = "Unable to find a good bracket for effect_inf"
        warn(msg)
    return bracket


def estimate_effect(moment_vector,
                    rc_range,
                    effect_segments):
    """Estimates effect and its gradient."""
    # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    # pylint: disable=invalid-name
    ntab = 10
    nmax = 10
    con = 1.4
    con2 = con * con
    big = sys.float_info.max
    safe = 2.0
    h = 1.0e-1
    errmax = 0.0
    effect_estimate = np.zeros((2, len(moment_vector)+1))
    deps = np.zeros(len(moment_vector))
    dmoments = np.zeros(len(moment_vector))
    a = np.zeros((ntab, ntab))
    fac = geop(con2, con2, ntab - 1)
    errt = np.zeros(ntab-1)
    # Get rc_inf and effect_inf
    rc_inf = rcinf(moment_vector)
    effect_inf = effectinf(moment_vector)
    # Check to make sure that rc_inf is not in rc_range.  If so,
    # effect is completely unidentified.
    if rc_range[0] <= rc_inf <= rc_range[1]:
        effect_estimate[0, 0] = -np.inf
        effect_estimate[1, 0] = np.inf
        effect_estimate[:, 1:] = 0.0
        return effect_estimate
    # IMPORTANT_effectS is a list of effect values for which rc(effect) needs
    # to be calculated. We don't know in advance how many important values
    # there will be, so we make IMPORTANT_effectS way too big, and initialize
    # it to all zeros (this choice is arbitrary).
    # Get simplified moments
    simplified_moments = simplify_moments(moment_vector)
    # k is the number of actual important effect values in IMPORTANT_effectS
    important_effects = np.array([])
    k = 1
    # Go piece by piece through effect_segments
    for i in range(1, len(effect_segments)):
        # Get the next pair of effects.  This represents a range of effects to
        # check
        current_effect_range = effect_segments[i-1:i+1]
        # Skip ahead to the next pair if effect_inf is in the current range
        if ((not np.isfinite(effect_inf)) or
           (current_effect_range[0] >= effect_inf) or
           (current_effect_range[1] <= effect_inf)):
            # Otherwise, calculate the range of rcs associated with that
            # range of effects
            current_rc_range = rcfast(current_effect_range,
                                      simplified_moments)
            # For each of the values in rc_range
            for j in range(1, 3):
                # See if that value satisfies rc(effect)-rc(j)=0 for
                # some effect in current_effect_range
                if (rc_range[j-1] > min(current_rc_range)) and \
                   (rc_range[j-1] < max(current_rc_range)):
                    # If so, find effect such that rc(effect)-rc(j)=0
                    # and put it inour list of IMPORTANT_effectS.  Of course,
                    # we can't quite find the exact effect.
                    tmp = zbrent(rc_minus_rc,
                                 current_effect_range[0],
                                 current_effect_range[1],
                                 1.0e-200,
                                 np.insert(simplified_moments,
                                           0,
                                           rc_range[j-1]))
                    important_effects = np.append(important_effects, tmp)
                    k = k + 1
    # Add effect_SEGMENTS to the list of IMPORTANT_effectS
    important_effects = np.append(important_effects, effect_segments)
    # Add the OLS effect to the list of IMPORTANT_effectS?
    # simplified_moments(3)/simplified_moments(2)
    # Calculate rc(effect) for every effect in IMPORTANT_effectS
    rc_segments = rcfast(important_effects, simplified_moments)
    # INRANGE = True if a particular value of effect satisfies the condition
    #     rc_range(1) <= rc(effect) <= rc_range(2)
    # Notice that we have put a little error tolerance in here, since
    # zbrent won't find the exact root.
    inrange = ((rc_segments >= rc_range[0]-0.001) &
               (rc_segments <= rc_range[1]+0.001))
    if k > 1:
        inrange[0:k-1] = True
    # If no IMPORTANT_effectS are in range, the identified set is empty
    if sum(inrange) == 0:
        effect_estimate[0, 0] = np.nan
    # If the lowest value in IMPORTANT_effectS is in range, then there is no
    # (finite) lower bound
    elif inrange[np.argmin(important_effects)]:
        effect_estimate[0, 0] = -np.inf
    else:
        # Otherwise the the lower bound for effect is the minimum value in
        # IMPORTANT_effectS that is in range
        effect_estimate[0, 0] = min(important_effects[inrange])
    # If no IMPORTANT_effectS are in range, the identified set is empty
    if sum(inrange) == 0:
        effect_estimate[1, 0] = np.nan
    # If the highest value in IMPORTANT_effectS is in range, then there is no
    # (finite) upper bound
    elif inrange[np.argmax(important_effects)]:
        effect_estimate[1, 0] = np.inf
    else:
        # Otherwise the the upper bound for effect is the maximum value in
        # IMPORTANT_effectS that is in range
        effect_estimate[1, 0] = max(important_effects[inrange])
    # Now we find the gradient
    # Take the gradient at both effect_L and effect_H
    for j in range(1, 3):
        effect = effect_estimate[j-1, 0]
        # The gradient can only be calculated if effect is finite.
        # This was hopefully caught above but check just in case.
        if not np.isfinite(effect):
            # If effect is infinite, then the gradient is zero.
            effect_estimate[j - 1, 1:] = 0.0
            continue

        # Gradients are estimated using a simple finite central difference:
        #            df/dx = (f(x+e)-f(x-e))/2e
        # where e is some small step size.  The tricky part is getting the
        # right step size.  The algorithm used here is an adaptation of
        # dfridr in Numerical Recipes.  However, that algorithm needs an
        # input initial step size h.
        #
        # http://www.fizyka.umk.pl/nrbook/c5-7.pdf: "As a function of input
        # h, it is typical for the accuracy to get better as h is made
        # larger, until a sudden point is reached where nonsensical
        # extrapolation produces early return with a large error. You
        # should therefore choose a fairly large value for h, but monitor
        # the returned value err, decreasing h if it is not small. For
        # functions whose characteristic x scale is of order unity, we
        # typically take h to be a few tenths."
        # So we try out starting values (h) until we get one that gives an
        # acceptable estimated error.
        for n in range(1, nmax+1):
            # Our candidate initial stepsize is 0.1, 0.001, ...
            h = 0.1 ** n
            # Initialize errmax
            errmax = 0.0
            # Initialize the finite-difference vector
            deps[:] = 0.0
            # First, we calculate the scalar-as-vector (drc / deffect)
            # hh is the current step size.
            hh = h
            # Calculate an approximate derivative using stepsize hh
            a[0, 0] = (rcfast(effect + hh, simplified_moments) -
                       rcfast(effect - hh, simplified_moments)) / \
                      (2.0 * hh)
            # Set the error to very large
            err = big
            # Now we try progressively smaller stepsizes
            for k in range(2, ntab + 1):
                # The new stepsize hh is the old stepsize divided by 1.4
                hh = hh / con
                # Calculate an approximate derivative with the new
                # stepsize
                a[0, k-1] = ((rcfast(effect + hh, simplified_moments) -
                              rcfast(effect - hh, simplified_moments)) /
                             (2.0 * hh))
                # Then use Neville's method to estimate the error
                for m in range(2, k + 1):
                    a[m - 1, k - 1] = ((a[m - 2, k - 1] *
                                        fac[m - 2] -
                                        a[m - 2, k - 2]) /
                                       (fac[m - 2] - 1.0))
                errt[0:k - 1] = np.maximum(abs(a[1:k, k - 1] -
                                               a[0:k - 1, k - 1]),
                                           abs(a[1:k, k - 1] -
                                               a[0:k - 1, k - 2]))
                ierrmin = np.nanargmin(errt[0:k - 1]) if \
                    any(np.isfinite(errt[0:k - 1])) else 0
                # If the approximation error is lower than any previous,
                # use that value
                if errt[ierrmin] <= err:
                    err = errt[ierrmin]
                    dfridr = a[1 + ierrmin, k - 1]
                if abs(a[k - 1, k - 1] - a[k - 2, k - 2]) >= (safe * err):
                    break
            # errmax is the biggest approximation error so far for the
            # current value of h
            errmax = max(errmax, err)
            # Now we have a candidate derivative drc/deffect
            deffect = dfridr
            # Second, estimate the vector (drc / dmoment_vector)
            for i in range(1, len(moment_vector) + 1):
                hh = h
                deps[i-1] = hh
                a[0, 0] = ((rcfun(moment_vector + deps, effect) -
                            rcfun(moment_vector - deps, effect)) /
                           (2.0 * hh))
                err = big
                for k in range(2, ntab + 1):
                    hh = hh / con
                    deps[i-1] = hh
                    a[0, k - 1] = (rcfun(moment_vector + deps,
                                         effect) -
                                   rcfun(moment_vector - deps,
                                         effect)) / (2.0 * hh)
                    for m in range(2, k + 1):
                        a[m - 1, k - 1] = (a[m - 2, k - 1] * fac[m - 2] -
                                           a[m - 2, k - 2]) / \
                                           (fac[m - 2] - 1.0)
                    errt[0:k - 1] = np.maximum(abs(a[1:k, k - 1] -
                                                   a[0:k - 1, k - 1]),
                                               abs(a[1:k, k - 1] -
                                                   a[0:k - 1, k - 2]))
                    ierrmin = np.nanargmin(errt[0:k - 1]) if \
                        any(np.isfinite(errt[0:k - 1])) else 0
                    if errt[ierrmin] <= err:
                        err = errt[ierrmin]
                        dfridr = a[1 + ierrmin, k - 1]
                    if abs(a[k - 1, k - 1] - a[k - 2, k - 2]) >= \
                       (safe * err):
                        break
                # errmax is the biggest approximation error so far for the
                # current value of h
                errmax = max(errmax, err)
                dmoments[i - 1] = dfridr
                deps[i - 1] = 0.0
            # At this point we have estimates of the derivatives stored in
            # deffect and dmoments. We also have the maximum approximation
            # error for the current h stored in errmax. If that
            # approximation error is "good enough" we are done and can
            # exit the loop
            if errmax < 0.01:
                break
            # Otherwise we will try again with a smaller h
            if n == nmax:
                msg1 = "Inaccurate SE for effectL/H."
                msg2 = "Try normalizing variables."
                warn(msg1 + " " + msg2)
        # Finally, we apply the implicit function theorem to calculate the
        # gradient that we actually need:
        #   deffect/dmoments = -(drc/dmoments)/(drc/deffect)
        effect_estimate[j-1, 1:] = -dmoments / deffect
    return effect_estimate


def simplify_moments(moment_vector):
    """Converts moment_vector into the six moments needed for the model."""
    # Get sizes
    # pylint: disable=invalid-name
    m = len(moment_vector)
    k = int((np.sqrt(1 + 8 * m) + 1) / 2)
    assert 2*(m + 1) == k ** 2 + k
    mvtmp = np.append(1.0, moment_vector)
    xtmp = np.empty((k, k))
    # The array XTMP will contain the full cross-product matrix E(WW')
    # where W = [1 X Y Z]
    h = 0
    for i in range(0, k):
        for j in range(i, k):
            xtmp[i, j] = mvtmp[h]
            xtmp[j, i] = mvtmp[h]
            h = h + 1
    assert h == (m + 1)
    # The array XX will contain the symmetric matrix E(XX')
    XX = xtmp[0:(k - 2), 0:(k - 2)]
    # The array XY will contain the vector E(XY)
    XY = xtmp[(k - 2), 0:(k - 2)]
    # The array XZ will contain the vector E(XZ)
    XZ = xtmp[(k - 1), 0:(k - 2)]
    # Now we fill in simplify_moments with the various moments.
    simplified_moments = np.full(6, float("nan"))
    # varY
    simplified_moments[0] = (moment_vector[m - 3] -
                             (moment_vector[k - 3]) ** 2)
    # varZ
    simplified_moments[1] = (moment_vector[m - 1] -
                             (moment_vector[k - 2]) ** 2)
    # covYZ
    simplified_moments[2] = (moment_vector[m - 2] -
                             moment_vector[k - 2]*moment_vector[k - 3])
    # The XX matrix could be singular, so catch that exception
    try:
        invXX = inv(XX)
        # varYhat
        simplified_moments[3] = (XY.T @ invXX @ XY -
                                 moment_vector[k - 3] ** 2)
        # varZhat
        simplified_moments[4] = (XZ.T @ invXX @ XZ -
                                 moment_vector[k - 2] ** 2)
        # covYZhat
        simplified_moments[5] = (XY.T @ invXX @ XZ -
                                 moment_vector[k - 2] * moment_vector[k - 3])
    except np.linalg.LinAlgError:
        # These values will return as NaN
        pass
    # When there is only one control variable, yhat and zhat are perfectly
    # correlated (positively or negatively) With rounding error, this can lead
    # to a correlation that is > 1 in absolute value.  This can create
    # problems, so we force the correlation to be exactly 1.
    # This also could happen if there is more than one control variable
    # but only one happens to have a nonzero coefficient.  I don't know how to
    # handle that case.
    if k == 4:
        simplified_moments[5] = (np.sign(simplified_moments[5]) *
                                 np.sqrt(simplified_moments[3] *
                                         simplified_moments[4]))
    return simplified_moments


def check_moments(moment_vector):
    """Checks that moment_vector is valid."""
    # pylint: disable=too-many-branches
    simplified_moments = simplify_moments(moment_vector)
    # First make sure that moment_vector describes a valid covariance matrix
    valid = True
    if not all(np.isfinite(simplified_moments)):
        valid = False
        if (all(np.isfinite(simplified_moments[0:3])) and
           all(np.isnan(simplified_moments[4:7]))):
            warn("Invalid data: nonsingular X'X matrix.")
        else:
            warn("Invalid data: unknown issue")
    if simplified_moments[0] < 0.0:
        valid = False
        warn(f"Invalid data: var(y) = {simplified_moments[0]} < 0")
    if simplified_moments[1] < 0.0:
        valid = False
        warn(f"Invalid data: var(z) = {simplified_moments[1]} < 0")
    if simplified_moments[3] < 0.0:
        valid = False
        warn(f"Invalid data: var(yhat) = {simplified_moments[3]} < 0")
    if simplified_moments[4] < 0.0:
        valid = False
        warn(f"Invalid data: var(zhat) = {simplified_moments[4]} < 0")
    if (np.abs(simplified_moments[2]) >
       np.sqrt(simplified_moments[0] * simplified_moments[1])):
        valid = False
        covyz = np.abs(simplified_moments[2])
        sdyz = np.sqrt(simplified_moments[0] * simplified_moments[1])
        msg1 = f"Invalid data: |cov(y,z)| = {covyz} "
        msg2 = f"> {sdyz} sqrt(var(y)*var(z))"
        warn(msg1 + msg2)
    # I'm not certain this condition can ever be triggered here
    if np.abs(simplified_moments[5]) > np.sqrt(simplified_moments[3] *
                                               simplified_moments[4]):
        valid = False
        covyz = np.abs(simplified_moments[5])
        sdyz = np.sqrt(simplified_moments[3] * simplified_moments[4])
        msg1 = f"Invalid data: cov(yh,zh) = {covyz}"
        msg2 = f" > {sdyz} sqrt(var(yh)*var(zh))"
        warn(msg1 + msg2)
    # Next make sure that the identifying conditions are satisfied.
    identified = valid
    if simplified_moments[0] == 0.0:
        identified = False
        warn("Model not identified: var(y) = 0")
    if simplified_moments[1] == 0.0:
        identified = False
        warn("Model not identified: var(z) = 0")
    if simplified_moments[3] == 0.0:
        identified = False
        warn("Model not identified: var(yhat) = 0")
    if simplified_moments[3] == simplified_moments[0]:
        identified = False
        warn("Model not identified: y is an exact linear function of X")
    # We may also want to check for var(zhat)=0.
    # The model is identified in this case, but we may need to take special
    # steps to get the calculations right.
    return valid, identified


def rcinf(moment_vector):
    """Calculates rc_inf."""
    simplified_moments = simplify_moments(moment_vector)
    # rc_inf is defined as sqrt( var(z)/var(zhat) - 1)
    # The check_moments subroutine should ensure that
    #   var(z) > 0 and that var(z) >= var(zhat) >= 0.
    # This implies that rc_inf >= 0.
    # Special values: If var(zhat) = 0, then rc_inf = +Infinity
    rc_inf = np.inf if simplified_moments[4] == 0.0 else \
        np.sqrt(np.maximum(simplified_moments[1] /
                           simplified_moments[4], 1.0) - 1.0)
    return rc_inf


def effectinf(moment_vector):
    """Calculates effect_inf."""
    simplified_moments = simplify_moments(moment_vector)
    # effect_inf is defined as
    #   cov(yhat,zhat)/var(zhat)
    # The check_moments subroutine should ensure that
    # var(zhat) >= 0 and that if var(zhat)=0 -> cov(yhat,zhat)=0.
    # Special values: If var(zhat)=0, then effect_inf = 0/0 = NaN.
    effect_inf = np.nan if simplified_moments[4] == 0.0 else \
        simplified_moments[5] / simplified_moments[4]
    return effect_inf


def rcfast(effect, simplified_moments):
    """Calculates rc for each effect in the given array."""
    # pylint: disable=invalid-name
    y = simplified_moments[0]
    z = simplified_moments[1]
    yz = simplified_moments[2]
    yhat = simplified_moments[3]
    zhat = simplified_moments[4]
    yzhat = simplified_moments[5]
    effect = np.atleast_1d(effect)
    lf0_num = (yhat -
               2.0 * effect * yzhat +
               effect ** 2 * zhat)
    lf0_denom = (y - yhat -
                 (2.0) * effect * (yz - yzhat) +
                 effect ** 2 * (z - zhat))
    lf1_num = (yz - yzhat - effect * (z - zhat))
    lf1_denom = (yzhat - effect * zhat)
    msk = ((lf0_denom != 0.0) &
           (lf1_denom != 0.0) &
           (np.sign(lf0_num) == np.sign(lf0_denom)))
    rc_fast = np.full(len(effect), np.nan)
    rc_fast[msk] = ((lf1_num[msk]/lf1_denom[msk]) *
                    np.sqrt(lf0_num[msk]/lf0_denom[msk]))
    return rc_fast


def negative_rcfast(effect, simplified_moments):
    """Calcualtes -rc(effect)."""
    return -rcfast(effect, simplified_moments)


def rcfun(moment_vector, effect):
    """Calculates rc(effect)."""
    rc_fast = rcfast(effect, simplify_moments(moment_vector))
    return rc_fast


def rc0_fun(moment_vector):
    """"Calculates rc(0)."""
    # rc0 is defined as:
    # (cov(y,z)/cov(yhat,zhat)-1) / sqrt(var(y)/var(yhat)-1)
    # The check_moments subroutine should ensure that
    #  var(y) >= var(yhat) > 0, so the denominator is
    # always positive and finite.
    # Special values: If cov(yhat,zhat)=0, then rc0 can
    #   be +Infinity, -Infinity, or NaN depending on the sign
    #   of cov(y,z).
    simplified_moments = simplify_moments(moment_vector)
    var_y = simplified_moments[0]
    cov_yz = simplified_moments[2]
    var_yhat = simplified_moments[3]
    cov_yzhat = simplified_moments[5]
    msk = ((var_y != var_yhat) &
           (cov_yzhat != 0.0) &
           (np.sign(var_yhat) == np.sign((var_y - var_yhat))))
    rcval = (((cov_yz - cov_yzhat)/cov_yzhat) *
             np.sqrt(var_yhat/(var_y - var_yhat))) if msk else np.nan
    return rcval


def rc_minus_rc(effect, simplified_moments_and_rc):
    """Calculates lamba(effect)."""
    rc1 = rcfast(effect, simplified_moments_and_rc[1:])
    rc0 = simplified_moments_and_rc[0]
    return rc1 - rc0


def estimate_parameter(func, moment_vector):
    """Estimates a parameter and its gradient."""
    # pylint: disable=too-many-locals,invalid-name
    parameter_estimate = np.zeros(len(moment_vector) + 1)
    parameter_estimate[0] = func(moment_vector)
    nmax = 10
    ntab = 10
    con = 1.4
    con2 = con ** 2
    h = 1.0e-4
    safe = 2.0
    big = 1.0e300
    deps = np.zeros(len(moment_vector))
    errt = np.zeros(ntab - 1)
    fac = geop(con2, con2, ntab - 1)
    a = np.zeros((ntab, ntab))
    if np.isfinite(parameter_estimate[0]):
        for n in range(1, nmax + 1):
            h = 0.1 ** n
            errmax = 0.0
            # We are estimating the gradient, i.e., a vector of derivatives
            # the same size as moment_vector
            for i in range(1, len(moment_vector) + 1):
                # Re-initialize DEPS
                deps[:] = 0.0
                # HH is the step size.  It is chosen by an algorithm borrowed
                # from the dfridr function in Numerical Recipes.  We start
                # with HH set to a predetermined value H.  After that, each
                # successive value of HH is the previous value divided by CON
                # (which is set to 1.4)
                hh = h
                # Set element i of DEPS to HH.
                deps[i - 1] = hh
                # Calculate the first approximation
                a[0, 0] = (func(moment_vector + deps) -
                           func(moment_vector - deps)) / (2.0*hh)
                dfridr = a[0, 0]   # WORKAROUND
                # The error is assumed to be a big number
                err = big
                # Try a total of NTAB different step sizes
                for j in range(2, ntab + 1):
                    # Generate the next step size
                    hh = hh / con
                    # Set DEPS based on that step size
                    deps[i - 1] = hh
                    # Calculate the approximate derivative for that step size
                    a[0, j - 1] = (func(moment_vector + deps) -
                                   func(moment_vector - deps)) / (2.0*hh)
                    # Next we estimate the approximation error for the current
                    # step size
                    for k in range(2, j + 1):
                        a[k - 1, j - 1] = (a[k - 2, j - 1] *
                                           fac[k - 2] -
                                           a[k - 2, j - 2]) / \
                                          (fac[k - 2] - 1.0)
                    errt[0:j - 1] = np.maximum(np.abs(a[1:j, j - 1] -
                                                      a[0:j - 1, j - 1]),
                                               np.abs(a[1:j, j - 1] -
                                                      a[0:j - 1, j - 2]))
                    ierrmin = np.nanargmin(errt[0:j - 1]) if \
                        any(np.isfinite(errt[0:j - 1])) else 0
                    # If the error is smaller than the lowest previous error,
                    # use that hh
                    if errt[ierrmin] <= err:
                        err = errt[ierrmin]
                        dfridr = a[1 + ierrmin, j - 1]
                    # If the error is much larger than the lowest previous
                    # error, stop
                    if np.abs(a[j - 1, j - 1] - a[j - 2, j - 2]) >= \
                       (safe * err):
                        break
                errmax = max(errmax, err)
                parameter_estimate[i] = dfridr
            if errmax < 0.01:
                break
            if n == nmax:
                msg1 = f"Inaccurate SE for {func.__name__}."
                msg2 = "Try normalizing variables."
                warn(msg1 + " " + msg2)
    else:
        parameter_estimate[1:] = 0.0   # or change to internal_nan
    return parameter_estimate


# Standard numerical algorithms


def brent(ax, bx, cx, func, tol, xopt):
    """Maximizes by Brent algorithm."""
    # pylint: disable=too-many-arguments,too-many-locals
    # pylint: disable=too-many-branches,too-many-statements
    # pylint: disable=invalid-name
    itmax = 1000
    cgold = 0.3819660
    zeps = 1.0e-3 * np.finfo(float).eps  # NOT SURE THIS WILL WORK
    a = min(ax, cx)
    b = max(ax, cx)
    v = bx
    w = v
    x = v
    e = 0.0
    fx = func(x, xopt)
    fv = fx
    fw = fx
    # NOTE: I've added the extraneous line below so that code-checking
    # tools do not flag the "e = d" statement below as referencing
    # a nonexistent variable. In practice, this statement will never
    # be reached in the first loop iteration, after which point d will be
    # defined.
    d = e
    for i in range(1, itmax + 1):
        xm = 0.5 * (a + b)
        tol1 = tol * abs(x) + zeps
        tol2 = 2.0 * tol1
        if abs(x - xm) <= (tol2 - 0.5 * (b - a)):
            brent_solution = x
            break
        if abs(e) > tol1:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)
            if q > 0.0:
                p = -p
            q = abs(q)
            etemp = e
            e = d     # See NOTE above
            if (abs(p) >= abs(0.5 * q * etemp)) or \
               (p <= q * (a - x)) or \
               (p >= q * (b - x)):
                e = (a - x) if (x >= xm) else (b - x)
                d = cgold * e
            else:
                d = p / q
                u = x + d
                if (u - a < tol2) or (b - u < tol2):
                    d = tol1 * np.sign(xm - x)
        else:
            e = (a - x) if (x >= xm) else (b - x)
            d = cgold * e
        u = (x + d) if (abs(d) >= tol1) else (x + tol1 * np.sign(d))
        fu = func(u, xopt)
        if fu <= fx:
            if u >= x:
                a = x
            else:
                b = x
            v = w
            w = x
            x = u
            fv = fw
            fw = fx
            fx = fu
        else:
            if u < x:
                a = u
            else:
                b = u
            if (fu <= fw) or (w == x):
                v = w
                fv = fw
                w = u
                fw = fu
            elif (fu <= fv) or (v == x) or (v == w):
                v = u
                fv = fu
    if i == itmax:
        brent_solution = x
        write_to_logfile("Brent exceeded maximum iterations.\n")
    return brent_solution


def zbrent(func, x1, x2, tol, xopt):
    """Finds a root using the Brent algorithm."""
    # pylint: disable=too-many-arguments,too-many-locals
    # pylint: disable=too-many-branches,too-many-statements
    # pylint: disable=consider-swap-variables
    # pylint: disable=invalid-name
    itmax = 1000
    eps = np.finfo(float).eps   # in fortran was epsilon(x1)
    a = x1
    b = x2
    fa = func(a, xopt)
    fb = func(b, xopt)
    if (((fa > 0.0) and (fb > 0.0)) or ((fa < 0.0) and (fb < 0.0))):
        write_to_logfile("Error in zbrent: Root is not bracketed")
        # call die("root must be bracketed for zbrent") # UPDATE
    c = b
    fc = fb
    for i in range(1, itmax + 1):
        if (((fb > 0.0) and (fc > 0.0)) or ((fb < 0.0) and (fc < 0.0))):
            c = a
            fc = fa
            d = b - a
            e = d
        if abs(fc) < abs(fb):
            a = b
            b = c
            c = a
            fa = fb
            fb = fc
            fc = fa
        # check for convergence
        tol1 = 2.0 * eps * abs(b) + 0.5 * tol
        xm = 0.5 * (c - b)
        if (abs(xm) <= tol1) or (fb == 0.0):
            zbrent_solution = b
            break
        if (abs(e) >= tol1) and (abs(fa) > abs(fb)):
            s = fb / fa
            if a == c:
                p = 2.0 * xm * s
                q = 1.0 - s
            else:
                q = fa / fc
                r = fb / fc
                p = s * (2.0 * xm * q * (q - r) - (b - a) * (r - 1.0))
                q = (q - 1.0) * (r - 1.0) * (s - 1.0)
            if p > 0.0:
                q = -q
            p = abs(p)
            if 2.0 * p < min(3.0 * xm * q - abs(tol1 * q), abs(e * q)):
                e = d
                d = p / q
            else:
                d = xm
                e = d
        else:
            d = xm
            e = d
        a = b
        fa = fb
        b = (b + d) if (abs(d) > tol1) else (b + tol1 * np.sign(xm))
        fb = func(b, xopt)
    if i == itmax:
        zbrent_solution = b
        write_to_logfile("zbrent: exceeded maximum iterations")
    return zbrent_solution


def geop(first, factor, nobs):
    """Creates a geometric series."""
    geometric_series = np.zeros(nobs)
    if nobs > 0:
        geometric_series[0] = first
    for obs in range(1, nobs):
        geometric_series[obs] = geometric_series[obs - 1] * factor
    return geometric_series


def get_column_names(arr, default_names=None):
    """Returns column names for an array_like object."""
    cols = default_names
    if isinstance(arr, pd.DataFrame):
        cols = arr.columns.tolist()
    elif isinstance(arr, pd.Series):
        cols = arr.name
    elif hasattr(arr, "design_info"):
        cols = arr.design_info.column_names
    return cols


def bkouter(arrow, msk):
    """Returns a vector's outer product with itself, as a vector."""
    return np.outer(arrow, arrow).flatten()[msk]


def check_rc(rc_range):
    """Checks that rc_range is valid."""
    assert isinstance(rc_range, np.ndarray)
    if rc_range.ndim != 1:
        msg1 = "rc_range should be 1-d array"
        msg2 = f" and is a {rc_range.ndim}-d array."
        raise TypeError(msg1 + msg2)
    if rc_range.shape[0] != 2:
        msg1 = "rc_range should have 2 elements"
        msg2 = f" and has {rc_range.shape[0]} element(s)."
        raise TypeError(msg1 + msg2)
    if rc_range.shape[0] != 2:
        msg1 = "rc_range should have 2 elements"
        msg2 = f" and has {rc_range.shape[0]} element(s)."
        raise TypeError(msg1 + msg2)
    if any(np.isnan(rc_range)):
        msg = "rc_range cannot be NaN."
        raise ValueError(msg)
    if rc_range[0] > rc_range[1]:
        msg1 = f"elements of rc_range ({rc_range})"
        msg2 = " must be in (weakly) ascending order."
        raise ValueError(msg1 + msg2)


def check_endog(endog):
    """Checks that endog is valid."""
    assert isinstance(endog, np.ndarray)
    if endog.ndim != 2:
        msg1 = "endog should be 2-d array"
        msg2 = f" and is a {endog.ndim}-d array."
        raise TypeError(msg1 + msg2)
    if endog.shape[1] != 2:
        msg1 = "endog should have 2 columns"
        msg2 = f"and has {endog.shape[1]} column(s)."
        raise TypeError(msg1 + msg2)


def check_exog(exog, nrows):
    """Checks that exog matrix is valid."""
    assert isinstance(exog, np.ndarray)
    if exog.ndim != 2:
        msg = f"exog should be 2-d array; is a {exog.ndim}-d array."
        raise TypeError(msg)
    if exog.shape[1] < 2:
        msg1 = "exog should have at least 2 columns"
        msg2 = f" and has {exog.shape[1]} column(s)."
        raise TypeError(msg1 + msg2)
    if exog.shape[0] != nrows:
        msg1 = f"endog has {nrows} rows"
        msg2 = f" and exog has {exog.shape[0]} rows."
        raise TypeError(msg1 + msg2)
    if any(exog[:, 0] != 1.0):
        msg = "first column of exog must be an intercept"
        raise ValueError(msg)


def check_covinfo(cov_type, vceadj):
    """Checks that cov_type and vceadj are valid."""
    if cov_type not in ("nonrobust", "cluster"):
        msg = f"cov_type '{cov_type}' not yet supported."
        raise ValueError(msg)
    if not isinstance(vceadj, (float, int)):
        msg = f"vceadj must be a number, is a {type(vceadj)}."
        raise TypeError(msg)
    if vceadj < 0.:
        msg = f"vceadj = {vceadj}, must be non-negative."
        raise ValueError(msg)


def check_ci(cilevel, citype=None):
    """Checks that cilevel and citype are valid."""
    if not isinstance(cilevel, (float, int)):
        msg = f"cilevel must be a number, is a {type(cilevel)}."
        raise TypeError(msg)
    if cilevel < 0.:
        msg = f"cilevel = {cilevel}, should be between 0 and 100."
        raise ValueError(msg)
    if citype is None:
        return None
    if citype not in ("conservative", "upper", "lower", "Imbens-Manski"):
        msg = f"Unsupported CI type {citype}."
        raise ValueError(msg)
    return None


def check_weights(weights, nrows):
    """Checks that weights are valid."""
    if weights is None:
        return None
    weights = np.asarray(weights)
    if weights.ndim != 1:
        msg1 = "weights must be a 1-d array"
        msg2 = f" but is a {weights.ndim}-d array."
        raise TypeError(msg1 + msg2)
    if not np.issubdtype(weights.dtype, np.number):
        msg1 = "weights must be an array of numbers"
        msg2 = f" but is an array of {weights.dtype}."
        raise TypeError(msg1 + msg2)
    if len(weights) != nrows:
        msg = f"len(weights) = {len(weights)} but nrows = {nrows}."
        raise TypeError(msg)
    if not all(np.isfinite(weights)):
        msg = "all weights must be finite."
        raise ValueError(msg)
    if sum(weights) <= 0:
        msg = "weights must sum to a positive number."
        raise ValueError(msg)
    return None


def robust_cov(dat,
               groupvar=None,
               weights=None):
    """Estimates cluster-robust covariance metrix"""
    resid = pd.DataFrame(dat - np.average(dat, weights=weights, axis=0))
    if weights is None:
        nobs = len(dat)
        weights = np.ones(nobs)/nobs
    else:
        nobs = sum(weights > 0.)
        weights = weights/sum(weights)
    umat = (np.asarray(resid).T * weights).T
    if groupvar is None:
        ubarmat = umat
        dofadj = nobs/(nobs-1)
    else:
        ubar = resid.groupby(groupvar).sum()
        ngroups = sum(pd.Series(weights).groupby(groupvar).sum() > 0)
        bothu = resid.join(ubar,
                           on=groupvar,
                           how="right",
                           rsuffix=".ubar").sort_index()
        ubarmat = np.asarray(bothu.loc[:,
                                       bothu.columns.str.endswith('.ubar')])
        ubarmat = (ubarmat.T * weights).T
        dofadj = ngroups / (ngroups - 1)
    out = np.dot(ubarmat.T, umat) * dofadj
    return out


# I/O and system functions for Stata


def get_command_arguments(args):
    """Retrieves command arguments, usually from sys.argv."""
    # ARGS should be a list of 1 to 5 strings like sys.argv
    if isinstance(args, list) and all(isinstance(item, str) for item in args):
        if len(args) > 5:
            msg = f"Unused program arguments {args[5:]}"
            warnings.warn(msg)
    else:
        msg = f"Invalid command arguments, using defaults: {args}"
        warnings.warn(msg)
        args = []
    _infile = args[1].strip() if len(args) > 1 else "in.txt"
    _outfile = args[2].strip() if len(args) > 2 else "pout.txt"
    _logfile = args[3].strip() if len(args) > 3 else "plog.txt"
    _detail_file = args[4].strip() if len(args) > 4 else ""
    return _infile, _outfile, _logfile, _detail_file


def set_logfile(fname):
    """Sets name of log file."""
    global LOGFILE  # pylint: disable=global-statement
    if isinstance(fname, str) or fname is None:
        LOGFILE = fname
    else:
        pass


def get_logfile():
    """Retrieves the name of the log file."""
    return LOGFILE


def write_to_logfile(msg, mode="a"):
    """Writes a note to the log file."""
    logfile = get_logfile()
    if logfile is None:
        return None
    try:
        with open(logfile,
                  mode,
                  encoding="utf-8") as log_file:
            log_file.write(msg)
    except OSError:
        fail_msg = f"Cannot write to logfile {logfile}."
        warnings.warn(fail_msg)
    return None


def start_logfile(logfile):
    """Starts the log file."""
    set_logfile(logfile)
    write_to_logfile(f"Log file {logfile} for RCR version 1.0\n",
                     mode="w")
    start_time = datetime.now().strftime("%H:%M on %m/%d/%y")
    write_to_logfile(f"Run at {start_time}.\n")


def read_data(infile):
    """Reads RCR data from infile."""
    # pylint: disable=too-many-statements
    write_to_logfile(f"Reading data from input file {infile}.\n")
    # infile argument should be a single string
    if not isinstance(infile, str):
        msg = "Infile should be a single string"
        die(msg)
    try:
        # Line 1 should be three whitespace delimited numbers
        line1 = pd.read_csv(infile,
                            delim_whitespace=True,
                            skiprows=[1, 2],
                            header=None).values[0, ]
        n_moments, n_rc, external_big_number = tuple(line1)
        # Line 2 should be n_moments whitespace delimited numbers
        moment_vector = pd.read_csv(infile,
                                    delim_whitespace=True,
                                    skiprows=[0, 2],
                                    header=None).values[0, ].astype(np.float64)
        # Lines 3+ should be two whitespace delimited numbers each
        rc_range = pd.read_csv(infile,
                               delim_whitespace=True,
                               skiprows=[0, 1],
                               header=None).values[0, ].astype(np.float64)
    except FileNotFoundError:
        msg = f"infile {infile} not found.\n"
        die(msg)
    except ValueError:
        msg = f"Incorrect format in infile {infile}.\n"
        die(msg)
    else:
        msg1 = f"Line 1: n_moments = {n_moments}, n_rc = {n_rc}"
        msg2 = f"external_big_number = {external_big_number}.\n"
        write_to_logfile(msg1 + ", " + msg2)
        mv_len = len(moment_vector)
        msg = f"Line 2: moment_vector = a vector of length {mv_len}.\n"
        write_to_logfile(msg)
        write_to_logfile(f"Line 3: rc_range = {rc_range}.\n")
        write_to_logfile("For calculations, rc_range,...\n")
        write_to_logfile(f"Data successfully loaded from file {infile}\n")
    # reset n_moments and n_rc if needed
    n_rc = int(n_rc)
    n_moments = int(n_moments)
    external_big_number = float(external_big_number)
    if n_moments != len(moment_vector):
        msg1 = f"n_moments reset from {n_moments} "
        msg2 = f"to len(moment_vector) = {len(moment_vector)}."
        warn(msg1 + msg2)
        n_moments = len(moment_vector)
    if len(rc_range) != 2*n_rc:
        true_n_rc = int(len(rc_range)/2)
        msg1 = f"n_rc reset from {n_rc} "
        msg2 = f"to len(rc_range)/2 = {true_n_rc}."
        warn(msg1 + msg2)
        n_rc = true_n_rc
    check_input_values(n_moments, n_rc, external_big_number)
    return n_moments, n_rc, external_big_number, \
        moment_vector, rc_range


def check_input_values(n_moments, n_rc, external_big_number):
    """Makes sure read_data has read in valid data."""
    # Check to make sure n_moments is a valid value
    #   1. It should be the same as the length of moment_vector.  if not,
    #      just reset it.
    #   2. It must be at least 9 (i.e., there must be at least one explanatory
    #      variable)
    assert n_moments >= 9
    #   3. The number of implied explanatory variables must be an integer
    k = int((np.sqrt(9 + 8 * n_moments) - 1) / 2)
    assert (2 * (n_moments + 1)) == int(k ** 2 + k)
    # Check to make sure n_rc is a valid (i.e., positive) value
    #   1. It should be positive.
    assert n_rc > 0
    #   2. For now, it should be one.
    assert n_rc == 1
    # Check to make sure external_big_number is a valid value
    assert external_big_number > 0.0
    # If external_big_number is bigger than sys.float_info.max, then issue a
    # warning but don't stop program. I'm not satisfied with this.
    if external_big_number > sys.float_info.max:
        msg1 = f"Largest Python real ({sys.float_info.max}) "
        msg2 = f"is less than largest in Stata {external_big_number}"
        warn(msg1 + msg2)


def write_results(result_matrix, outfile):
    """Writes the results_matrix array to outfile."""
    write_to_logfile(f"Writing results to output file {outfile}.\n")
    write_to_logfile("Actual results = ...\n")
    try:
        with np.printoptions(threshold=np.inf, linewidth=np.inf):
            np.savetxt(outfile, result_matrix, delimiter=" ")
    except OSError:
        msg = f"Cannot write to output file {outfile}."
        warn(msg)
    else:
        write_to_logfile("RCR successfully concluded.\n")


def write_details(effectvec, rcvec, detail_file):
    """Outputs effectvec and rcvec to _detail_file."""
    if len(detail_file) > 0:
        try:
            with open(detail_file,
                      mode="w",
                      encoding="utf-8") as d_file:
                d_file.write("theta, lambda \n")
                for i, effect in enumerate(effectvec):
                    d_file.write(f"{effect}, {rcvec[i]} \n")
        except OSError:
            warn(f"Cannot write to detail file {detail_file}.")


def warn(msg):
    """Issues warning (to logfile and python warning system) but continues."""
    write_to_logfile("WARNING: " + msg + "\n")
    warnings.warn(msg)


def die(msg):
    """Writes message to log file and raises exception."""
    write_to_logfile("FATAL ERROR: " + msg)
    raise RuntimeError(msg)


def translate_result(mat, inf=np.inf, nan=np.nan):
    """Translates inf and NaN values (e.g., for passing to Stata)."""
    newmat = np.copy(mat)
    msk1 = np.isinf(newmat)
    newmat[msk1] = np.sign(newmat[msk1])*inf
    msk2 = np.isnan(newmat)
    newmat[msk2] = nan
    return newmat


def stata_exe(argv):
    """
    Performs tasks needed by Stata RCR command.

    Stata will call this file with the command:

       python rcrbounds.py [infile outfile logfile]

       where

           infile      An optional argument giving the name
                       of the input file.  Default is IN.TXT.

           outfile     An optional argument giving the name of
                       the output file. Default is OUT.TXT.

           logfile     An optional argument giving the name of
                       the output file. Default is LOG.TXT.

       This function will read in the INFILE, perform
       the calculations, and then write the results to OUTFILE.
       The program may also report information on its status to
       LOGFILE.
    """
    (infile0, outfile0, logfile0,
        detail_file0) = get_command_arguments(argv)

    # Start the log file
    start_logfile(logfile0)

    # Read in the data from INFILE
    (external_big_number0, moment_vector0,
        rc_range0) = read_data(infile0)[2:5]

    # Perform the calculations and put the results in result_matrix
    (result_matrix0, effectvec0, rcvec0) = estimate_model(moment_vector0,
                                                          rc_range0)

    # Write out the data to OUTFILE
    write_results(translate_result(result_matrix0,
                                   inf=external_big_number0,
                                   nan=0.0),
                  outfile0)

    if detail_file0 != "":
        write_details(effectvec0, rcvec0, detail_file0)

    # Close the log file
    set_logfile(None)


#############################################################################
# Begin run code
#############################################################################

# If this program is called as a script, it will be from the Stata RCR command.
# Call stata_exe to perform the actions required for this purpose.
if __name__ == "__main__":
    stata_exe(sys.argv)

#############################################################################
# End run code
#############################################################################
