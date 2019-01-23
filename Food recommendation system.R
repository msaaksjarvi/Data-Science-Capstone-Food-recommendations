#Loading libraries
library(readr)
library(tidyverse)
library(caret)
library(tidyr)
library(ggplot2)

#Get data
Reviews <- read_csv("~/Desktop/Reviews.csv")

#Define scores and ProductId as numeric
Reviews$Score <- as.numeric(Reviews$Score)
Reviews$ProductId <- as.factor(Reviews$ProductId)
Reviews$ProductId <- as.numeric(Reviews$ProductId)

#Create train and test sets, with 10% of the data allocated to the test set
set.seed(1)
test_index <- createDataPartition(y = Reviews$Score, times = 1, p = 0.1, list = FALSE)
r_train <- Reviews[-test_index,]
r_test <- Reviews[test_index,]

#Make sure ProductId and UserId in the test set are also in the train set
Reviewtest <- r_test %>%
  semi_join(r_train, by = "ProductId") %>%
  semi_join(r_train, by = "UserId")

#Add rows from removed test set back into the train set
removed <- anti_join(r_test, Reviewtest)
r_train <- rbind(r_train, removed)

#Remove files that are no longer needed
rm(Reviews, test_index, r_test, removed)

#Examine the structure of the training set
head(r_train)

#Examine the dimensions of the dataset
dim(r_train)
summary(r_train)

#Numbers of users and products
r_train %>%
  summarize(n_users = n_distinct(UserId),
            n_products = n_distinct(ProductId))

#Distribution of users
r_train %>%
  count(UserId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "blue") +
  scale_x_log10() +
  ggtitle("Users")

#Distribution of products
r_train %>%
  count(ProductId) %>%
  ggplot(aes(n)) +
  geom_histogram(bins = 30, color = "blue") +
  scale_x_log10() +
  ggtitle("Products")

#Distribution of ratings
ratings <- as.vector(r_train$Score)
ratings <- factor(ratings)
qplot(ratings) +
  ggtitle("Distribution of ratings")

#Mean and sd of review scores
mean(r_train$Score)
sd(r_train$Score)

#Top 10 rated products
r_train %>% group_by(ProductId) %>%
  summarize(n = n()) %>%
  arrange(desc(n))

#RMSE metric
RMSE <- function(true_ratings, predicted_ratings) {
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#Predicting results from average rating
mu <- mean(r_train$Score)
model_1_rmse <- RMSE(Reviewtest$Score, mu)
rmse_results <- data_frame(method = "Just the average", RMSE = model_1_rmse)
rmse_results %>% knitr::kable()

#Including product bias
mu <- mean(r_train$Score)
product_avgs <- r_train %>%
  group_by(ProductId) %>%
  summarize(b_i = mean(Score- mu))

predicted_ratings <- mu + Reviewtest %>%
  left_join(product_avgs, by="ProductId") %>%
  .$b_i

#Estimating RMSE for product bias
model_2_rmse <-RMSE(predicted_ratings, Reviewtest$Score)
rmse_results <- bind_rows(rmse_results, data_frame(method= "Product Effect Model", RMSE = model_2_rmse))
rmse_results %>% knitr::kable()


#Including user and product bias
user_avgs <- r_train %>%
  left_join(product_avgs, by="ProductId") %>%
  group_by(UserId) %>%
  summarize(b_u = mean(Score - mu - b_i))

predicted_ratings <- Reviewtest %>%
  left_join(product_avgs, by="ProductId") %>%
  left_join(user_avgs, by="UserId") %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred

#Estimating RMSE for user and product bias
model_3_rmse <- RMSE(predicted_ratings, Reviewtest$Score)
rmse_results <- bind_rows(rmse_results, data_frame(method = "Product + User Effects Model", RMSE = model_3_rmse))

rmse_results %>% knitr::kable()

#Switching to regularization, estimating lambda
lambda <- 3
mu <- mean(r_train$Score)
food_reg_avgs <- r_train %>%
  group_by(ProductId) %>%
  summarize(b_i = sum(Score - mu)/(n()+lambda), n_i = n())

#Plot of regularized estimates vs. least square estimates
data_frame(original = product_avgs$b_i,
           regularized = food_reg_avgs$b_i,
           n = food_reg_avgs$n_i) %>%
  ggplot(aes(original, regularized, size =sqrt(n))) +
  geom_point(shape = 1, alpha = 0.5, color = "black")

predicted_ratings <- Reviewtest %>%
  left_join(food_reg_avgs, by="ProductId") %>%
  mutate(pred = mu + b_i) %>%
  .$pred

#Estimating RMSE for regularization
model_4_rmse <- RMSE(predicted_ratings, Reviewtest$Score)
rmse_results <- bind_rows(rmse_results, data_frame(method = "Regularization", RMSE = model_4_rmse))
rmse_results %>% knitr::kable()     

#Choosing the optimal lambda via cross-fertilization
lambdas <- seq(1, 10, 0.25)
mu<- mean(r_train$Score)
just_the_sum <- r_train %>%
  group_by(ProductId) %>%
  summarize(s = sum(Score-mu), n_i = n())

rmses <- sapply(lambdas, function(l) {
  predicted_ratings <- Reviewtest %>%
    left_join(just_the_sum, by="ProductId") %>%
    mutate(b_i=s/(n_i+l)) %>%
    mutate(pred = mu + b_i) %>%
    .$pred
  return(RMSE(predicted_ratings, Reviewtest$Score))
})
#Plotting lambdas
qplot(lambdas, rmses)

#Optimal lambda
l <- lambdas[which.min(rmses)]
l

#Including product and user effects into the regularized model (Regularization+)
  b_i <- r_train %>%
    group_by(ProductId) %>%
    summarize(b_i = sum(Score - mu)/(n()+l))
  
  b_u <- r_train %>%
    left_join(b_i, by="ProductId") %>%
    group_by(UserId) %>%
    summarize(b_u = sum(Score - b_i- mu)/(n()+l))
  
  predicted_ratings <- Reviewtest %>%
    left_join(b_i, by="ProductId") %>%
    left_join(b_u, by="UserId") %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred

#Estimating RMSE for regularization+ model
model_5_rmse <- RMSE(predicted_ratings, Reviewtest$Score)
rmse_results <- bind_rows(rmse_results, data_frame(method = "Regularization+", RMSE = model_5_rmse))
rmse_results %>% knitr::kable()  


