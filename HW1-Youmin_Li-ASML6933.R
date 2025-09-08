# abe6933 asml - HW1 - Youmin Li
##library
library(ggplot2)
library(dplyr)
library(tibble)
## 1.1 prediction with inputs threshold, 1 if higher than 0.5 or 0
setwd('/Users/macpro/Desktop/Youmin-phd/machine learning/[1].assignments/hw1')
hw1 <- read.csv("SML.NN.data.csv")
summary(hw1)
train <- subset(hw1, set == "train")
valid <- subset(hw1, set == "valid")
test <- subset(hw1, set == "test")
x_train <- as.matrix(train[, c("X1", "X2")])
y_train <- train$Y
##euclidean distance
getClass1Prop <- function(x, r) {
  dists <- sqrt(rowSums((x_train - rep(x, each=nrow(x_train)))^2))
  within_r <- dists <= r
  if (!any(within_r)) {
    return(NA)
  }
  prop <- mean(y_train[within_r] == 1)
  return(prop)
}
##getClass1Prop(x, r) >= 0.5

##1.2compute the misclassification rate
misclassRate <- function(data, r) {
  x_mat <- as.matrix(data[, c("X1", "X2")])
  y_true <- data$Y
  n <- nrow(data)
  preds <- numeric(n)
  for (i in 1:n) {
    p1 <- getClass1Prop(x_mat[i, ], r)
    preds[i] <- ifelse(!is.na(p1) && p1 >= 0.5, 1, 0)
  }
  misclass_rate <- mean(preds != y_true)
  return(misclass_rate)
}

##1.3explore train data and valid data
p_train <- ggplot(train, aes(x = X1, y = X2, color = Y)) +
  geom_point(alpha = 0.8, size = 2) +
  coord_equal() +
  labs(
    title = "Figure 1. Training Data: X1 vs X2",
    x = "X1",
    y = "X2",
    color = "Y"
  ) +
  theme_minimal(base_size = 12)
print(p_train)
p_valid <- ggplot(valid, aes(x = X1, y = X2, color = Y)) +
  geom_point(alpha = 0.8, size = 2) +
  coord_equal() +
  labs(
    title = "Figure 1. Valid Data: X1 vs X2",
    x = "X1",
    y = "X2",
    cat = "color"
  ) +
  theme_minimal(base_size = 12)
print(p_valid)

##1.4 misclassification rate r*
r_values <- seq(0.01, 1.00, by=0.01)
val_errors <- sapply(r_values, function(r) misclassRate(valid, r))
min_err <- min(val_errors, na.rm=TRUE)
r_star <- r_values[which.min(val_errors)]
cat("Lowest validation misclassification rate:", min_err, "at r* =", r_star, "\n")

##1.6 fixed radius vs fixed number of neighbors K approach
test_error <- misclassRate(test, r_star)
cat("Test misclassification rate with r* =", r_star, "is", test_error, "\n")


##2.1 normal distribution
mu_x <- 0; mu_y <- 0
sx   <- 1; sy   <- 1
rho  <- 0.5
dbvnorm <- function(x, y, mu_x=0, mu_y=0, sx=1, sy=1, rho=0) {
  z1 <- (x - mu_x)/sx
  z2 <- (y - mu_y)/sy
  denom <- 2*pi*sx*sy*sqrt(1 - rho^2)
  expo  <- -1/(2*(1 - rho^2)) * (z1^2 - 2*rho*z1*z2 + z2^2)
  exp(expo) / denom
}

fx_num <- function(x) {
  out <- integrate(function(y) dbvnorm(x, y, mu_x, mu_y, sx, sy, rho),
                   lower = -Inf, upper = Inf,
                   rel.tol = 1e-8, abs.tol = 0)
  out$value
}
x_grid <- seq(-3, 3, length.out = 201)

fx_hat <- sapply(x_grid, fx_num)
fx_std <- dnorm(x_grid, mean = mu_x, sd = sx)

df_plot <- tibble(
  x = x_grid,
  fx_num = fx_hat,
  fx_norm01 = fx_std
)

ggplot(df_plot, aes(x)) +
  geom_point(aes(y = fx_num), color = "red", size = 1.2, alpha = 0.7) +
  geom_line(aes(y = fx_norm01), color = "blue", linewidth = 1) +
  labs(
    title = "Figure 2.1: Marginal f_X(x) vs N(0,1)",
    subtitle = expression(mu[x] == 0 * "," ~ mu[y] == 0 * "," ~ 
                            sigma[x] == 1 * "," ~ sigma[y] == 1 * "," ~ rho == 0.5),
    x = "x",
    y = "Density"
  ) +
  theme_minimal(base_size = 12)

##3.1MLE
set.seed(0)
x <- rexp(100, rate = 10)   
loglik <- function(lambda) {
  if (lambda <= 0) return(-Inf)  
  100 * log(lambda) - lambda * sum(x)
}
res <- optimize(loglik, interval=c(0.0001, 50), maximum=TRUE)
res$maximum  
res$objective  

## 4 CI
B <- 1000 
n <- 4
sigma <- 1
z <- qnorm(0.975)
t <- qt(0.975, df=3)
cover1 <- logical(B)

for (j in 1:B) {
  set.seed(j)                          
  x <- rnorm(n, mean = 0, sd = sigma) 
  xbar <- mean(x)
  half_len <- z * sigma / sqrt(n)   
  ci <- c(xbar - half_len, xbar + half_len)  
  cover1[j] <- (ci[1] <= 0 && 0 <= ci[2])    
}

cov_case1 <- mean(cover1)
cat(sprintf("Case 1 (known sigma): empirical coverage = %.3f\n", cov_case1))

cover2 <- logical(B)  
width2 <- numeric(B)    
for (j in 1:B) {
  set.seed(j)              
  x <- rnorm(n, 0, 1)        
  xbar <- mean(x)
  s <- sd(x)                 
  half_len <- t * s / sqrt(n)
  ci <- c(xbar - half_len, xbar + half_len)  
  cover2[j] <- (ci[1] <= 0 && 0 <= ci[2])    
  width2[j] <- diff(ci)                      
}

cov_case2 <- mean(cover2)
cat(sprintf("Case 2 (t-CI, df=3): empirical coverage = %.3f\n", cov_case2))

hist(width2,
     breaks = 30,
     main   = "Case 2: Histogram of CI widths (t-based, n=4, df=3)",
     xlab   = "Interval width")

cover3 <- logical(B)   
width3 <- numeric(B) 

for (j in 1:B) {
  set.seed(j)               
  x <- rnorm(n, 0, 1)       
  xbar <- mean(x)
  s <- sd(x)
  half_len <- z * s / sqrt(n)
  ci <- c(xbar - half_len, xbar + half_len) 
  cover3[j] <- (ci[1] <= 0 && 0 <= ci[2])   
  width3[j]  <- diff(ci)                      
}

cov_case3 <- mean(cover3)
cat(sprintf("Case 3 (z-CI with s, n=4): empirical coverage = %.3f\n", cov_case3))

hist(width3,
     breaks = 30,
     main   = "Case 3: Histogram of CI widths (z-based, n=4)",
     xlab   = "Interval width")
