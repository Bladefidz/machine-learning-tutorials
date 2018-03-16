# Install all packages used in this tutorial
install.packages('C50')
install.packages('gmodels')
install.packages('mlr')

# Importing the dataset
dataset = read.csv('dataset/car-evaluation/car.csv')

# Analyzing data
library(ggplot2)
# Plot buying histogram
ggplot(data=dataset,
       aes(x=buying)) +
  geom_bar() +
  ggtitle("Histogram buying") +
  facet_wrap(~car)
# Plot maintenance histogram
ggplot(data=dataset,
       aes(x=maint)) +
  geom_bar() +
  ggtitle("Histogram maintenance") +
  facet_wrap(~car)
# Plot doors histogram
ggplot(data=dataset,
       aes(x=doors)) +
  geom_bar() +
  ggtitle("Histogram doors") +
  facet_wrap(~car)
# Plot persons histogram
ggplot(data=dataset,
       aes(x=persons)) +
  geom_bar() +
  ggtitle("Histogram persons") +
  facet_wrap(~car)
# Plot lug_boots histogram
ggplot(data=dataset,
       aes(x=lug_boots)) +
  geom_bar() +
  ggtitle("Histogram lugage boots") +
  facet_wrap(~car)
# Plot safety histogram
ggplot(data=dataset,
       aes(x=safety)) +
  geom_bar() +
  ggtitle("Histogram safety") +
  facet_wrap(~car)

# Encode label as factor
dataset$car = factor(dataset$car, levels = c('unacc', 'acc', 'good', 'vgood'))

# Split label set into training set and test set with ratio 75:25
library(caTools)
split = sample.split(dataset$car, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Aplly C50 algorithm
library(C50)
decTree = C5.0(formula = car ~ ., data = training_set)

# Making prediction using model created by C50 algorithm
y_pred = predict(decTree, newdata = test_set[-7], type = 'class')

# Plot the tree
plot(decTree)

# View model's summary
summary(decTree)

# Evaluate performance using Confusion Matrix
cm = table(test_set[, 7], y_pred)

# More detailed confusion matrix
library(gmodels)
gmodels::CrossTable(test_set$car,
           y_pred,
           prop.chisq = FALSE,
           prop.c     = FALSE,
           prop.r     = FALSE,
           dnn = c('actual default', 'predicted default'))

# Produce learning curves to evaluate our splitting policy
library(mlr)
dataset.task = makeClassifTask(id = "car", data = dataset, target = "car")
learningCurve = generateLearningCurveData(
  learners = c("classif.C50"),
  task = dataset.task,
  percs = seq(0.1, 1, by = 0.1),
  resampling = makeResampleDesc(method = "CV", iters = 5, predict = 'both'),
  measures = list(setAggregation(acc, train.mean), setAggregation(acc, test.mean)),
  show.info = TRUE)
plotLearningCurve(learningCurve, facet = 'learner')