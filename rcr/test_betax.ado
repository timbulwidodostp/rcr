version 8
capture program drop test_betax

program define test_betax, rclass
	syntax [=/exp] 
	tempname cmd
	local `cmd' = e(cmd)
	if ("``cmd''" != "rcr") {
		error 301
	}
	if "`exp'"=="" {
		local exp 0
	}
	tempname tL tH tmax tdelt pvalue
	scalar `tL' = (_b[betaxL]-`exp')/_se[betaxL]
	scalar `tH' = (`exp'-_b[betaxH])/_se[betaxH]
	scalar `tmax' = max(`tL',`tH')
	scalar `tdelt' = (_b[betaxH]-_b[betaxL])/max(_se[betaxL],_se[betaxH])
	scalar `pvalue' = 1 - normal(`tmax' + `tdelt') + normal(-`tmax')
	di _newline as text " ( 1)     " as result "betax = `exp'" _newline _newline _column(10) as text "P-value = " as result %6.4f `pvalue'
	return scalar p = `pvalue'
end program

* test_betax

