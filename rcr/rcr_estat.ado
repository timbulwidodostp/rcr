version 8
capture program drop rcr_estat

program define rcr_estat
	gettoken key rest : 0, parse(", ")
	local lkey = length(`"`key'"')

	if "`e(cmd)'" == "" {
		error 301
	}

	if `"`key'"' == "," { 
		dis as err "subcommand expected" 
		exit 198
	}
	
	if `"`key'"' == "ic" {
		IC `rest'
	}
	else if `"`key'"' == substr("summarize",1,max(2,`lkey')) {
		SUmmarize `rest'
	}
	else if `"`key'"' == "vce" {
		VCE `rest'
	}
	else if `"`key'"' == substr("bootstrap",1,max(4,`lkey')) {
		di as text "WARNING: Bootstrap not yet tested with RCR model. Use at your own risk."
		_bs_display `rest'
	}
	else if `"`key'"' == `""' {
		di as error "subcommand required"
		exit 321
	}
	else {
		dis as err `"invalid subcommand `key'"'
		exit 321
	}
end


// default handlers

program IC
	dis as err "IC option not supported with RCR model"
	exit 321
end


program SUmmarize
	local depvar = e(depvar)
	local treatvar = e(treatvar)
	local ctrlvar = e(ctrlvar)
	estat_summ `depvar' `treatvar' `ctrlvar' `0'
end


program VCE
	vce `0'
end


