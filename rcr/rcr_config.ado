capture program drop rcr_config
program define rcr_config, rclass
	syntax [, forceos(string) forceversion(string) forcereq(string) nopython]
	di _newline "CONFIGURATION INFORMATION FOR RCR PACKAGE"
	di "------------------------------------------"
	di "The current RCR package is written in Python and requires that your"
	di "Stata is set up with Python integration.  This command will help you"
	di "insure that your setup is compatible." _newline
	
	local version = c(stata_version)
	if "`forceversion'" == "" {
		local version = c(stata_version)
	}
	else {
		local version "`forceversion'"
	}
	if "`forceos'" == "" {
		local os = c(os)
	}
	else {
		local os "`forceos'"
	}
	if "`forcereq'" == "" {
		local requirements "numpy pandas scipy"
	}
	else {
		local requirements "`forcereq'"
	}
	local py = 1
	
	if `py' == 1 {
		di "STEP 1: Does your Stata version support Python?"
		if (`version' >= 16){
			di "  YES, your Stata version (`version') supports Python" _newline
		}
		else {
			di "  NO, your Stata version (`version') does not support Python"
			di "  Python support is available in Stata versions 16 and above." _newline
			local py = 0
		}
	}

	if `py' == 1 {
		di "STEP 2: Is your Stata version linked to a compatible Python installation?"
		capture python : 1
		if _rc == 0 & "`python'" == "" {
			local python_exec = c(python_exec)
			di "  YES, your Stata version is linked to a compatible Python installation"
			di "  (`python_exec')." _newline
		}
		else { 
			di "  NO, your Stata version is not linked to a compatible Python installation." _newline
			di "    To fix this:
			di "    1. Use {help python:python search} to see if you have"
			di "       a compatible version of Python available on your computer."
			di "    2. If you do not have a compatible version of Python available,"
			di "       you can install one from {browse www.anaconda.com/products/distribution} "
			di "    3. Once you have a compatible version of Python installed,"
			di "       you may need to use {help python: python set exec} to tell Stata where it is." _newline
			local py = 0
		}
	}

	if `py' == 1 {
		di "STEP 3: Does your Python installation include all required modules?"
		local missing_modules ""
		foreach module in `requirements' {
			capture python which `module'
			if _rc {
				local missing_modules "`module' `missing_modules'"
			}
		}
		if "`missing_modules'" == "" {
			di "  YES, your Python installation includes all required modules." _newline
		}
		else {
			di "  NO, your Python installation is missing the following required modules."
			di "  `missing_modules'" _newline
			local py = 0
		}
	}

	if `py' == 1 {
		di "RCR will use the latest (Python) version." _newline
		di "Add the exe(windows-fortran) or exe(unix-fortran) optional argument"
		di "if you want to use the older (Fortran) version"
		local default_version "python"
	}
	else if (`py' == 0) & "`os'" == "Windows" {
		di "RCR will use the older (Fortran) version."
		local default_version "windows-fortran"
	}
	else if (`py' == 0) & "`os'" == "Unix" {
		di "RCR will use the older (Fortran) version."
		local default_version "unix-fortran"
	}
	else if (`py' == 0) & "`os'" == "MacOSX" {
		di "RCR is not supported on MacOSX for Stata versions < 16."
		local default_version "none"
	}
	else if (`py' == 0) {		
		di "RCR is not supported on unknown operating system: `os'"
		local default_version "none"
	}
	return local default_version "`default_version'"
end
