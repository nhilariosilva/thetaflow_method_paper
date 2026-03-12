# Install and load required package
library(survival)

# Load the dataset
data <- read.csv("tcga_brca_count_data.csv")

# Inspect the data (optional)
head(data)
str(data)

# Create the survival object
surv_obj <- Surv(time = data$time/365, event = data$delta)

# Fit a Weibull survival model (parametric)
weibull_model <- survreg(surv_obj ~ 1, data = data, dist = "weibull")

# Print model summary
summary(weibull_model)

# Extract Weibull parameters
shape <- 1 / weibull_model$scale
scale <- exp(coef(weibull_model))

# Create time grid for smooth curve
t_grid <- seq(0, max(data$time/365), length.out = 200)

# Weibull survival function
weibull_surv <- exp(-(t_grid / scale)^shape)

# Kaplan-Meier estimator
km_fit <- survfit(surv_obj ~ 1)

# Plot Kaplan-Meier curve
plot(km_fit,
     xlab = "Time",
     ylab = "Survival Probability",
     main = "Kaplan-Meier vs Weibull Survival Curve",
     col = "black",
     lwd = 2,
     conf.int = FALSE)

# Add Weibull predicted survival curve
lines(t_grid, weibull_surv, col = "red", lwd = 2)

# Add legend
legend("topright",
       legend = c("Kaplan-Meier", "Weibull fit"),
       col = c("black", "red"),
       lwd = 2)
