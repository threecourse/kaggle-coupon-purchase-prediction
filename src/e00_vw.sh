# -------------------------------------------------------------------------
# This file runs vowpal wabbit, train with train data and predict test data.
# -------------------------------------------------------------------------

# train model by vowpal wabbit
# logistic regression with L1 regularization
# train using 90% of the data, 10% of the data is used for watchlist (by default setting)
cd ../model
vw -d train.vwtxt -c -k -P 10000000 --passes 200 -q AE -q AK -q AN --cubic DGH --cubic DGL -q BN -q DN --cubic AOP --cubic DGR -f train.vwmdl --loss_function logistic --l1 1e-8

# predict test data and write
vw -d test.vwtxt -t -i train.vwmdl -p predict.txt
