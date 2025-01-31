version 8
capture program drop rcr_predict

program define rcr_predict
	di as error "PREDICT not supported for RCR model"
	error 321
end
