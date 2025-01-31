version 8
capture program drop rcrplot
program define rcrplot
	syntax [, xrange(numlist max=2 min=2 ascending) yrange(numlist max=2 min=2 ascending) ]
	tempname cmd
	local `cmd' = e(cmd)
	if ("``cmd''" != "rcr") {
		error 301
	}
	capture confirm variable betax lambda
	if (_rc > 0) {
		local rc = _rc
		di as error "Information needed for rcrplot unavailable; use details option when calling rcr"
		error 321
	}
	preserve
	if "`xrange'" == "" {
		local xmin -50
		local xmax 50
	} 
	else {
		tokenize "`xrange'"
		local xmin `1'
		local xmax `2'
	}
	if "`yrange'" == "" {
		local ymin -50
		local ymax 50
	} 
	else {
		tokenize "`yrange'"
		local ymin `1'
		local ymax `2'
	}
	local betaxInf = _b[betaxInf]
	local lambdaInf = _b[lambdaInf]
	local lambdaL = e(lambdaL)
	local lambdaH = e(lambdaH)
	local betaxL = _b[betaxL]
	local betaxH = _b[betaxH]
	if c(stata_version) < 11 {
		local lambdasym "lambda"
		local betasym "beta"
		local Lambdasym "Lambda"
		local Betasym "Beta"
	} 
	else {
		local lambdasym "{&lambda}"
		local betasym "{&beta}{subscript:x}"
		local Lambdasym "[{&lambda}{superscript:L},{&lambda}{superscript:H}]"
		local Betasym "[{&beta}{subscript:x}{superscript:L},{&beta}{subscript:x}{superscript:H}]"	
	}
	quietly keep if betax > `xmin' & betax < `xmax' & lambda > `ymin' & lambda < `ymax'

	twoway line lambda betax if betax < _b[betaxInf]  || /*
	*/	line lambda betax if betax > _b[betaxInf] , lstyle(p1) || /*
	*/	line lambda betax if lambda > `lambdaL' & lambda < `lambdaH', lstyle(p1) lwidth(thick) || /*
	*/	scatteri `ymin' `betaxInf' `ymax' `betaxInf' , connect(l) msymbol(none) lpattern(dash)  || /*
	*/	scatteri `lambdaInf' `xmin' `lambdaInf' `xmax' , connect(direct) msymbol(none) lpattern(dash) || /*
	*/	scatteri `lambdaL' `xmin' `lambdaL' `betaxH' `ymin' `betaxH' `ymin' `betaxL' `lambdaH' `betaxL' `lambdaH' `xmin', connect(l) msymbol(none) lpattern(dot)  || /*
	*/	scatteri `lambdaL' `xmin' `lambdaH' `xmin'  , connect(l) msymbol(none) lwidth(thick)  || /*
	*/	scatteri `ymin' `betaxL' `ymin' `betaxH', connect(l) msymbol(none) lwidth(thick) ||  , /*
	*/	xtitle("Effect (`betasym')") ytitle("Relative correlation (`lambdasym')") legend(order(1 - " " 5 4 7 8 ) /*
	*/	label(1 "`lambdasym'(.) function") label(4 "`betasym'{superscript:{&infinity}}") label(5 "`lambdasym'{superscript:{&infinity}}") label(7 "`Lambdasym'") label(8 "`Betasym'"))
	restore
end

