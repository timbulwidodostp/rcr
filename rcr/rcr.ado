#delimit ;
/**********************************************************************
*
* RCR.ADO	Stata ado-file
*
* Description: Performs RCR estimation in Stata
*
* Authors:	Mohsen Javdani, Simon Fraser University
*		Brian Krauth, Simon Fraser University
*
*
*
***********************************************************************/

/**********************************************************************
* Main RCR program
***********************************************************************/
capture program drop rcr;
* set type double, permanently; /* I removed this because we don't want to make any persistent hidden changes to the user's environment */
program define rcr , eclass byable(recall) /* sortpreserve [I took this out because it generated an error msg with detail option */;
	
	/* If we are "replaying" (i.e. the user typed RCR with no arguments, or ESTIMATES REPLAY) just display the most recent results */
	if (replay()){;
		di_rcr `0';
		Footer_rcr `0';
		exit;
	};
	
	/* Otherwise, process all of the arguments */
	syntax varlist(min=3)/* First var = y(outcome) , Second var = x(treatment), Remaining vars= c(controls) */
								/* Limit on number of control variables is a function of Stata's limits on MATSIZE */
			[if] [in] [fw aw pw iw] /* Standard estimation command arguments, handled in the standard way */
			[, CLuster(varname) /* Standard option for cluster-corrected standard errors */
								/* We don't have the robust option - the program uses Stata's MEAN command, which doesn't support it */
			vce(namelist min = 2 max = 2) /* alternative to cluster */
			SAVe /* Undocumented option to save intermediate files */
			exe(string) /* Option to force a particular version */
			vceadj(real 1.0) 
			DETails /* Option to save details */
			citype(string) /* This is a special option for the type of confidence interval to calculate for BetaX */
			Level(cilevel) /* Standard option for specifying the confidence level */
			lambda(numlist max=2 min=2 missingokay ) ]; /* Lambda describes the lower and upper bound for the lambda (relative correlation) parameter */
														/* A missing value for the upper [lower] bound indicates that there is no upper [lower] bound */
													
	tempname lamb length moments lambf results V b gradient;
														
/***** (1) Process the command options ******/	
	/* Process varlist */
	/* The first variable name in varlist refers to the outcome variable and will be stored in DEPVAR */
	gettoken depvar indepvar: varlist;
	/* The second variable name in varlist refers to the treatment variable and will be stored in TREATVAR */
	/* The remaining variable name(s) refer to the control variable(s) and will be stored in CTRLVAR */
	gettoken treatvar ctrlvar: indepvar;
	/* The next line processes the IF and IN user options according to Stata's rules. */
	marksample touse;
	/* The weights (fw aw pw iw) and level are automatically processed by Stata. */
	/* Fill in some default values */
	if ("`citype'" == "") {;
		local citype "Conservative";
	};
	if ("`lambda'" == "") {;
		local lambda= "0 1"; /*The default interval is [0,1]*/
	};
	/* Move the input lambda into a matrix */
	local lambda : list retokenize lambda;
	local lambda = subinstr("(`lambda')"," ",",",.);
	matrix `lamb'=`lambda';
	/* If vceadj < 0, then issue an error */
	if `vceadj' < 0 {;
		di as error "Negative values for vceadj not allowed";
		error 111;
	};
	/* Process vce */
	if "`vce'" != "" {;
		gettoken vce1 vce2: vce;
		if "`vce1'" != "cluster" {;
			di as error "vce option `vce1' not allowed";
			error 198;
		};
		confirm variable `vce2';
		unab vce2 : `vce2';
		if "`cluster'" == "" {;
			local cluster "`vce2'";
		};
		if "`cluster'" != "`vce2'" {;
			di as error "options cluster() and vce(cluster) are in conflict";
			error 100;
		};
	};
/******** (2) Set the appropriate matsize **********/
	/*** The number of explanatory variables is limited by the value of matsize.  The algorithm is based on the second moment ***/
	/*** matrix of the data.  With K variables (1 outcome, 1 explanatory, K-2 control), that matrix is ***/
	/*** (K+1)*(K+1) because of the intercept.   So there can be no more than:  ***/
	/***      no more than 25 = floor(sqrt(800)-3) control variables for Stata BE/IC ***/
	/***      no more than 101 = floor(sqrt(11000)-3) control variables for Stata SE/MP ***/
	/*** NOTE: The second moment matrix is symmetric, so the code could be rewritten to take advantage of that.  ***/
	/*** By my calculation this would raise the Stata IC limit to 36 control variables, and the Stata SE limit to 146.  ***/
	local mat_max=c(max_matsize);
	local mat_min=c(min_matsize);
	local mat_current=c(matsize);
	local numall: word count `varlist';
	local mat_needed = (`numall' + 1)^2;
	if (`mat_needed' > `mat_max') {;
		di as error "Too many (" (`numall' - 2)	") explanatory variables. Maximum number for this Stata version is " (floor(sqrt(c(max_matsize)))-3);
		exit 103;
	};
	/* matsize does not apply for Stata version 16 and up */
	if (`mat_needed' > `mat_current') & (c(stata_version) < 16) {;
		set matsize `mat_needed';
	};
	
/***** (3) Calculate the moment vector (moments) and its variance (V) ******/	

	/* Grab the number of control variables*/
	local num : word count `ctrlvar'; 
	/* Grab the number of all the variables specified in the RCR command (should be num + 2) */
	local numall: word count `varlist'; 
	/* Check for whether there are enough observations */
	quietly count if `touse';
	local nobs = r(N);
	if `nobs' == 0 {;
		error 2000;
	};	
	if `nobs'  < `numall' {;
		error 2001;
	};
	/* Check to make sure the covariance matrix is positive definite */
	quietly correlate `varlist' if `touse', covariance;
	tempname covmat;
	matrix `covmat' = r(C);
	if (det(`covmat') <= 0) {;
		if `covmat'[1,1] <= 0 {;
			di as error "Error: Dependent variable is constant.";
			error 1;
		};
		if `covmat'[2,2] <= 0 {;
			di as error "Error: Treatment variable is constant.";
			error 1;
		};
		matrix `covmat' = `covmat'[3...,3...];
		if (diag0cnt(`covmat') > 0) {;
			di as error "Error: At least one control variable is constant.";
			error 1;
		};
		if (det(`covmat')) <= 0 {;
			di as error "Error: Control variables are perfectly collinear.";
			error 1;
		};
		di as error "Warning: At least one variable is perfectly collinear with other variables.";
		di as error "RCR will continue but results should be viewed with caution.";
	};
	/* Tokenize divides string into tokens, storing the result in `1', `2', ...
	   (the positional local macros).  Tokens are determined based on the
	   parsing characters pchars, which default to a space if not specified */
	tokenize `ctrlvar'; 
	/* Generate all the temporary variables we will need for the moment vector */
	forvalues i=1/`numall'{;
		forvalues j=1/`numall'{;
			tempvar X`i'X`j';
			capture tempvar X`i'Y;
			capture tempvar X`i'Z;
		};
	};
	tempvar y2 yz z2;
	/* 	Now, for each of the variables in the moment vector, we need to do two things:
		(1) Generate the variable if it doesn't already exist.
		(2) Maintain a running list of the variables we have generated, in the macro BIGLIST 
		
		WARNING: Be very careful with BIGLIST.  It can grow very large, and this creates a hidden
		risk in Stata.  The maximum size of a *macro* in Stata is very large (67,784 in Intercooled)
		but the maximum size of a *string variable* is very small (244).  What's worse
		is that if you use a macro in any string expression (roughly, anything that involves an equals
		sign) it will get quietly truncated to 244 characters.  No error or warning message.
		This can create very hard-to-find bugs in the program.
		
		So the rule of thumb is to never use BIGLIST in any expression involving a "=".
	*/
	/* First, add the original variables.  They don't need to be generated. */
	local biglist "`ctrlvar' `depvar' `treatvar'";
	/* Next, add all of the cross-products between X and (X,y,z) */
	forvalues i=1/`num'{;
			forvalues j=1/`num'{;
				if `i'<=`j'{;
					/*The following code assign values to the temp names created before, these are 
					X products with no repetition*/
					capture generate double `X`i'X`j''=``i''*``j''; /*``i'' and ``j'' come from elements of ctrlvar
									     that I tokenized above*/
					local biglist "`biglist' `X`i'X`j''"; /* adding X products to the local macro*/
				};
			};
			/* The following code assigns values to X(i)Y products, i changing from 1 to #ctrl vars*/
			capture generate double `X`i'Y'=``i''*`depvar';
			local biglist "`biglist' `X`i'Y'";
			/* The following code assigns values to X(i)Z products, i changing from 1 to #ctrl vars*/
			capture generate double `X`i'Z'=``i''*`treatvar';
			local biglist "`biglist' `X`i'Z'";
	};
	/* We also want the cross-products of y and z */
	generate double `y2'=`depvar'*`depvar';
	generate double `yz'=`depvar'*`treatvar';
	generate double `z2'=`treatvar'*`treatvar';
	local biglist "`biglist' `y2' `yz' `z2'"; /*since we want to replicate the same order of variables we have in
						    R code, we add `y2' `yz' `z2' at the very end*/
	/* This line is where we actually estimate the moment vector.  It is here and only here that
	   the if, in, weight, and cluster options are used. */
	quietly mean `biglist' if `touse' [`weight'`exp'], cluster(`cluster');

/***** (4) Prepare the results for output to the Fortran program******/	
	/* LENGTH will be the first row in the file.  It contains three numbers:
		The size of the moment vector (colsof(e(b))
		The number of lambda ranges provided (always 1 for this program)
		A big floating-point number.  All numbers above this will be treated as infinite (maxdouble()/10)
	*/
	matrix `length'=(colsof(e(b)),1,maxdouble()/10); /* this is the length of the moment vector and lambda vector*/	
	/* MOMENTS will be the second row in the file.  It is the vector of moments that we just calculated */
	matrix `moments' = e(b);
	/* LAMBF will be the third row in the file.  It replaces missing values (which Fortran can't handle) with very large values. */
	matrix `lambf' = ( cond(missing(`lamb'[1,1]),-maxdouble(),`lamb'[1,1]), cond(missing(`lamb'[1,2]),maxdouble(),`lamb'[1,2]));
	/* The following code will check that the number of integers specified in lambda option is even*/
	/* It's currently unnecessary, since we currently allow 2 and only 2.  We'll keep it here in case we change.
	local ncolslamb=colsof(lamb);
	tempvar remainder;
	generate double `remainder'=mod(`ncolslamb',2);
	if (`remainder'~=0) {;
		di as err "you have to specify an interval for lambda e.g. lambda(3,6)";
     	exit 198;
     };
     */  	                           
	
/***** (5) Output the moment vector and other information to a text file *****/
	/* The data will be passed in temporary files */
	tempfile input_file output_file log_file detail_file;
	if ("`save'" == "save") {;
		di "Files to be saved";
		local input_file "in.txt";  
		local output_file "out.txt";  
		local log_file "log.txt"; 
		local detail_file "details.txt"; 
	};
	if ("`details'" != "details") {;
		local detail_file "";
	};
	/* Write the data to the `input_file' file */
	quietly mat2txt, matrix(`length') saving("`input_file'") replace; /*vector of lengths*/
	quietly mat2txt, matrix(`moments') saving("`input_file'") append;/*vector of moments*/
	quietly mat2txt, matrix(`lambf') saving("`input_file'") append; /*lambda vector*/	

/***** (6) Call the RCR program *****/
	/* See if Python is supported in this environment */
	quietly rcr_config;
	if "`exe'" == "" {;
		local exe = r(default_version);
	};
	/***** the "RCR" should be stored in an ADO folder along with RCR program*************/
	/* Find the RCR program */
	if ("`exe'" == "python"){;
		quietly findfile "rcrbounds.py";
		local rcr_py = r(fn);
	};
	else if ("`exe'" == "windows-fortran") {;
		quietly findfile "rcr.exe";
		local rcr_exe = r(fn);
	};
	else if ("`exe'" == "unix-fortran") {;
		quietly findfile "rcr";
		local rcr_exe = r(fn);
	};
	else {;
		di "Executable `exe' is not supported.  Run rcr_config to check configuration.";
		return;
	};
	/* Input path(s) to libraries required by the Linux RCR executable (this is machine-specific) */
    /* The user can set the path by hand using the global variable rcr_path, or will be set to some
       standard Unix library locations */
    if "${rcr_path}" == "" {;
	    local path_to_libs "LD_LIBRARY_PATH=/lib64/:/usr/local/lib64/";
    };
    else {;
        local path_to_libs "LD_LIBRARY_PATH=$rcr_path";
    };
	/* Check to see if the output_file already exists */
	capture confirm file "`output_file'";
	/* Delete it if it does exist */
	if (_rc == 0) {;
		erase "`output_file'";	
	};
	/* Check to see if the log_file already exists */
	capture confirm file "`log_file'";
	/* Delete it if it does exist */
	if (_rc == 0) {;
		erase "`log_file'";	
	};
	/* Execute the RCR program.  */
	if ("`exe'" == "python"){;
		python script `rcr_py', args("`input_file'" "`output_file'" "`log_file'" "`detail_file'");
	};
	else if ("`exe'" == "windows-fortran") {;
		winexec "`rcr_exe'" "`input_file'" "`output_file'" "`log_file'" "`detail_file'";
	};
	else if ("`exe'" == "unix-fortran") {;
		shell `path_to_libs' `rcr_exe' `input_file' `output_file' `log_file' `detail_file'; /* quotes around local macros won't work in Linux shell! */
	};
	/* The following lines of code pauses the Stata program until the RCR program has ended.  */
	/* Check to see if the output_file exists */
	capture confirm file "`output_file'";
	/* Then repeat the following until it exists */
   	while _rc != 0 {;
   		/* Wait one second */
    	sleep 1000;
    	/* Check to see if the file is present */
    	capture confirm file "`output_file'";
   	};
   	
/***** (7) Read the text file output by the RCR program *****/
	/* Save the current state of the data, because we will be reading in a new file */
	preserve;
	/*First, we read in the Fortran output file to see if the Fortran program has encountered any error*/
	/*If Fortran encounters any problem, it will write the word "error" in the output file */
	/*We will read the output file and if it contains the word "error", the program will stop by displaying an error 
	and printing the log file on the screen for user to detect the error*/
	tempname outfile;
	file open `outfile' using "`log_file'", read;
	file read `outfile' line;
	while r(eof)==0 {;
		if (strpos(lower("`line'"),"error") > 0) {;
			di as error "Error in external program RCR.  Error message is:" _newline;
			di as text "`line'" _newline;
			di as error "Complete log file is : " _newline _newline;
			type "`log_file'";
			error 1;
		};
		if (strpos(lower("`line'"),"warning") > 0) {;
			di as error "`line'" _newline;
		};
		file read `outfile' line;
	};
	file close `outfile';
	/* Read in the data from the file FORTRAN produced*/
	quietly insheet using "`output_file'", delimiter(" ") double clear;
	/* Convert it to a matrix which is called "results" */
	mkmat _all, matrix(`results');
	/* Restore the original data set */
	restore;
	/* Get some data from the earlier estimation */
	matrix `V' = e(V);
	local nobs = e(N); /* NOBS is the number of observations used for the calculation */
	local Nclust = e(N_clust);/*number of clusters*/
	local ncolsb=colsof(`moments');/*this line grabs the number of columns in b matrix which is equivalent to the number of variables 
				in the moment vector, we need this when we read in the moment vector into FORTRAN*/
	matrix `b' = `results'[1..rowsof(`results'),1]'; /*this line grabs the first column of matrix "results" which are the
						     variables of interest FORTRAN program estimated*/
	matrix `gradient' = `results'[1..rowsof(`results'),2..(`ncolsb'+1)]; /*the rest of the columns of "results" matrix 
									are gradients*/ 
	matrix `V' = `vceadj'*`gradient' * `V' * (`gradient''); /*using delta method to calculate the variance-covariance matrix*/
	
	/* Put proper row/column names on b and V */
	matrix colnames `b' = lambdaInf betaxInf lambda0 betaxL betaxH;
	matrix rownames `V' = lambdaInf betaxInf lambda0 betaxL betaxH;
	matrix colnames `V' = lambdaInf betaxInf lambda0 betaxL betaxH;

/***** (6) Post results to e() *****/

	/* Clear out whatever's in there now */
	ereturn clear;
	/* Start by posting the new vector of parameter estimates and its covariance matrix */
	ereturn post `b' `V', depname("`depvar'") esample(`touse') obs(`nobs');
	
	/* Retrieve some information to display later */
	ereturn local title "RCR model";
	ereturn local estat_cmd "rcr_estat";
	ereturn local predict "rcr_predict";
	ereturn local depvar "`depvar'";
	ereturn local treatvar "`treatvar'";
	ereturn local ctrlvar "`ctrlvar'";
	ereturn scalar lambdaL=`lamb'[1,1]; 
	ereturn scalar lambdaH=`lamb'[1,2]; 
	ereturn local citype = proper("`citype'");
	ereturn scalar cilevel = `level';
	if (e(citype) == "Conservative") ci_conservative, level(`level');
	else if (e(citype) == "Lower") ci_lower, level(`level');
	else if (e(citype) == "Upper") ci_upper, level(`level');
	else if (e(citype) == "Imbens-Manski") ci_imbensmanski, level(`level');
	else di as error "WARNING: Unsupported citype `citype'";
	ereturn scalar betaxCI_L = r(betaxCI_L);
	ereturn scalar betaxCI_H = r(betaxCI_H);	
	if "`weight'" != "" {;
		ereturn local wexp "`exp'";
		ereturn local wtype "`weight'";
	};
	/* Only if cluster was specified... */
	if ("`cluster'" != "") {;
		ereturn scalar N_clust = `Nclust';
		ereturn local clustvar "`cluster'";
		ereturn local vcetype "Robust";		
	};
	/* This should be the very last thing added to e() */
	ereturn local cmd "rcr";
	
/***** (7) Print results *****/	

	di_rcr, level(`level'); /*refer to the program "di_rcr" below for more information on di_rcr program*/
	Footer_rcr, level(`level'); /*refer to the program "Footer" below for more information on Footer program*/

	/* Reset MATSIZE to its original value (not needed for Stata version 16 and up).  */
	if (c(stata_version) < 16) {;
		set matsize `mat_current';
	};
	if "`details'" == "details" {;
		quietly insheet using `detail_file', clear comma;
		rename theta betax;
		label variable betax "Assumed effect";
		label variable lambda "Implied relative correlation, i.e., lambda(betax)";
		label data "Detailed data generated by RCR command";
		rcrplot;
	};
	
end;




/**********************************************************************
* Helper functions 
***********************************************************************/

/*----------------------------------------------------------------------
* MAT2TXT
* 
* Description:	mat2txt outputs a Stata matrix to a text file for use 
*				in other programs such as word processors or spreadsheets.
*
* Source: Michael Blasnik (mblasnik@verizon.net) (M. Blasnik and Associates)
	  Ben Jann (ben.jann@soz.gess.ethz.ch) (ETH Zurich)
, with modifications.
------------------------------------------------------------------------*/
program define mat2txt;
	version 8.2;
	syntax , Matrix(name) SAVing(str) [ REPlace APPend Title(str) Format(str) NOTe(str) ];
	local format "%10.0g";
	local format "%16.0g";
	local formatn: word count `format';
	local saving "`saving'";
	tempname myfile;
	file open `myfile' using "`saving'", write text `append' `replace';
	local nrows=rowsof(`matrix');
	local ncols=colsof(`matrix');
	forvalues r=1/`nrows' {;
		local rowname: word `r' of `rownames';
		file write `myfile' /* `"`rowname'"' _tab */;
		forvalues c=1/`ncols' {;
			if `c'<=`formatn' local fmt: word `c' of `format';
			file write `myfile' `fmt' (`matrix'[`r',`c']) _tab;
		};
		file write `myfile' _n;
	};
	file close `myfile';
end;


/* this program displays the results on the screen*/
program di_rcr;
	syntax [,Level(cilevel)];
	di _n as text `"`e(title)'"' _col(55) `"Number of obs ="' as result %9.0g e(N);
	di as text _col(46) `"Lower bound on lambda  ="'  as result %9.0g e(lambdaL);
	di as text _col(46) `"Upper bound on lambda  ="'  as result %9.0g e(lambdaH);
	di _newline;
	ereturn display, plus level(`level');	
end;




/* The following program will calculate the confidence interval for betax and displays it at the bottom panel 
   of the estimation table along with the CI type*/
program Footer_rcr;
	syntax [, Level(cilevel)];
	if (e(citype) == "Conservative") ci_conservative, level(`level');
	if (e(citype) == "Lower") ci_lower, level(`level');
	if (e(citype) == "Upper") ci_upper, level(`level');
	if (e(citype) == "Imbens-Manski") ci_imbensmanski, level(`level');
	local betaxCI_L = r(betaxCI_L);
	local betaxCI_H = r(betaxCI_H);
	di as text %12s `"betax"' " {c |}" _skip(3) as text _skip "(" e(citype) _skip "confidence interval)"
	as result %9.0g _col(58) `betaxCI_L' _col(70) %9.0g `betaxCI_H';/*this creates the line that displays the CI for betax in the footer*/ 
		
	di as text in smcl "{hline 13}{c BT}{hline 64}";
	
	di as text `"Treatment Variable:"' as result _skip(3) e(treatvar);/* this will display the name of treatment variable*/

	local maxlen 57;
	local vlist = e(ctrlvar);
	local vlist = ltrim("`vlist'");	
	local header "as text `"Control Variables    :"'";
	foreach vname in `vlist' {;
		if (length("`current_line' `vname'") > `maxlen') {;
			di `header' _col(22) as result "`current_line'";
			local current_line " `vname'";
			local header "";
		};
		else {;
			local current_line "`current_line' `vname'";
		};
	};
	di `header' _col(22) as result "`current_line'";

	
end;
	
program define ci_conservative, rclass;
	syntax , Level(cilevel);
	if _se[betaxL] > 0 {;
		return scalar betaxCI_L = _b[betaxL]-(invnorm(1-((100-`level')/200.0))* _se[betaxL]); /*Lower bound of betax's CI*/
	}; else {;
		return scalar betaxCI_L = -maxdouble()/10;
	};
	if _se[betaxH] > 0 {;
		return scalar betaxCI_H = _b[betaxH]+(invnorm(1-((100-`level')/200.0))* _se[betaxH]); /*Upper bound of betax's CI*/
	}; else {;
		return scalar betaxCI_H = maxdouble()/10;
	};		
end;
program define ci_imbensmanski, rclass;
	syntax , Level(cilevel);
	tempname cv_min cv_max delta cv;
	/* The Imbens-Manski critical value lies somewhere between CV_MIN (the critical value for a one-tailed test) 
	   and CV_MAX (the critical value for a two-tailed test).  For example, if level=95, then CV_MIN = 1.64, CV_MAX= 1.96 */
	scalar `cv_min'=invnorm(1-((100-`level')/100.0));
	scalar `cv_max'=invnorm(1-((100-`level')/200.0));
	/* DELTA is the estimated size of the identified set, divided by its standard error */
	scalar `delta' = ((_b[betaxH])-(_b[betaxL]))/ max(_se[betaxL],_se[betaxH]);
	/* If either betax_H or betax_L is infinite, we essentially have a one-tailed CI: the critical value will be equal to CV_MIN */
	scalar `cv' = `cv_min';
	if ( !missing(`delta')) {;
	/* Otherwise, we need to calculate the critical value based on Imbens-Manski (2004), Econometrica Vol. 72*/
		while ((`cv_max' - `cv_min') > epsfloat()) {;
			scalar `cv' = 0.5*(`cv_min' + `cv_max');
			/* The NORM function was renamed in later Stata versions */
			version 8: if ((norm (`cv' + `delta' )  - norm( - `cv')) - (`level'/100) < 0) scalar `cv_min' = `cv';
			else scalar `cv_max' = `cv';				
		};
	};
	if _se[betaxL] > 0 {;
		return scalar betaxCI_L = _b[betaxL]-((`cv')* _se[betaxL]); /*Lower bound of betax's CI*/
	}; else {;
		return scalar betaxCI_L = -maxdouble()/10;
	};
	if _se[betaxH] > 0 {;
		return scalar betaxCI_H = _b[betaxH]+((`cv')* _se[betaxH]); /*Upper bound of betax's CI*/
	}; else {;
		return scalar betaxCI_H = maxdouble()/10;
	};		
end;

program define ci_lower, rclass;
	syntax , Level(cilevel);
	return scalar betaxCI_L = -maxdouble()/10;
	if _se[betaxH] > 0 {;
		return scalar betaxCI_H = _b[betaxH]+(invnorm(1-((100-`level')/100.0))* _se[betaxH]); /*Upper bound of betax's CI*/
	}; else {;
		return scalar betaxCI_H = maxdouble()/10;
	};		
end;
program define ci_upper, rclass;
	syntax , Level(cilevel);
	if _se[betaxL] > 0 {;
		return scalar betaxCI_L = _b[betaxL]-(invnorm(1-((100-`level')/100.0))* _se[betaxL]); /*Lower bound of betax's CI*/
	}; else {;
		return scalar betaxCI_L = -maxdouble()/10;
	};
	return scalar betaxCI_H = maxdouble()/10;
end;
