 
 clear all
 
 * Import data
 import delimited "C:\Users\feder\Dropbox\Github\industrial-organization-I\ps2\entryData.csv", encoding(Big5) clear 

 
 



* Create configuration identifier
egen config = concat(v5 v6 v7)
  

gen y = .

replace y = 1 if config == "000"
replace y = 2 if config == "100"
replace y = 3 if config == "110"
replace y = 4 if config == "101"
replace y = 5 if config == "111"
replace y = 6 if config == "010"
replace y = 7 if config == "011"
replace y = 8 if config == "001"

tab y 
* observe only 1 2 3 5


* Estimate Multinomial Logit (base category is 1)

mlogit y v1 v2 v3 v4,baseoutcome(1)

* predict probabilities
predict p1 p2 p3 p5

gen p4 = 0
gen p6 = 0
gen p7 = 0
gen p8 = 0

keep p*

export delimited using "C:\Users\feder\Dropbox\Github\industrial-organization-I\ps2\probs.csv", replace



