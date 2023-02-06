library(tidymodels)
library(visdat)
library(tidyr)
library(car)
library(pROC)
library(ggplot2)
library(vip)
library(rpart.plot)
library(DALEXtra)


path1<-"C:\\Users\\xyz\\Desktop\\vinit\\R project Data\\"
path1


hr_train<-read.csv(paste0(path1,"hr_train.csv"),sep = ',',stringsAsFactors = FALSE)

hr_test<-read.csv(paste0(path1,"hr_test.csv"),sep = ',',stringsAsFactors = FALSE)

glimpse(hr_train)
glimpse(hr_test)

setdiff(names(hr_train),names(hr_test))

dp_pipe=recipe(left~.,data=hr_train) %>% 
  update_role(sales,salary,new_role="to_dummies") %>% 
  step_rm(has_role("drop_vars")) %>% 
  step_unknown(has_role("to_dummies"),new_level="__missing__") %>% 
  step_other(has_role("to_dummies"),threshold =0.02,other="__other__") %>% 
  step_dummy(has_role("to_dummies")) %>%
  step_impute_median(all_numeric(),-all_outcomes())

dp_pipe=prep(dp_pipe)

sort(unique(hr_train$sales))
sort(unique(hr_train$salary))

train=bake(dp_pipe,new_data=NULL)
test=bake(dp_pipe,new_data=hr_test)

train$left <- as.factor(train$left)

tree_model=decision_tree(
  cost_complexity = tune(),
  tree_depth = tune(),
  min_n = tune()
) %>%
  set_engine("rpart") %>%
  set_mode("classification")


folds = vfold_cv(train, v = 5)

tree_grid = grid_regular(cost_complexity(), tree_depth(),
                         min_n(), levels = 3)

my_res=tune_grid(
  tree_model,
  left~ .,
  resamples = folds,
  grid = tree_grid,
  metrics = metric_set(roc_auc),
  control = control_grid(verbose = TRUE)
)


show_notes(my_res)

autoplot(my_res)+theme_light()

my_res %>% show_best()


final_tree_fit=tree_model %>% 
  finalize_model(select_best(my_res)) %>%
  fit(left~.,data=train)

rpart.plot(final_tree_fit$fit)

final_tree_fit %>%
  vip(geom = "col", aesthetics = list(fill = "midnightblue", alpha = 0.8)) +
  scale_y_continuous(expand = c(0, 0))

train_pred=predict(final_tree_fit,new_data = train,type="prob") %>% select(.pred_1)

test_pred=predict(final_tree_fit,new_data = test,type="prob") %>% select(.pred_1)

train.score=train_pred$.pred_1

real=train$left

rocit = ROCit::rocit(score = train.score, 
                     class = real) 

kplot=ROCit::ksplot(rocit,legend=F)

my_cutoff=kplot$`KS Cutoff`

test_hard_class=as.numeric(test_pred>my_cutoff)



test_pred=predict(final_tree_fit,new_data = test,type="prob") %>% select(.pred_1)
write.csv(test_pred,'Vinit_Pawar_P4_part2.csv',row.names = F)
