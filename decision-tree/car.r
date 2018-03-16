# Install all packages used in this tutorial
install.packages('C50')
install.packages('gmodels')
install.packages('mlr')

# Importing the dataset
dataset = read.csv('dataset/car-evaluation/car.csv')

# Analyzing attribute distribution of each criteria
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

# Apply randomized splitting data set into training set and test set
library(caTools)
split = sample.split(dataset$car, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Apply C50 algorithm on training set
library(C50)
decTree = C5.0(formula = car ~ ., data = training_set)
decTree

# Using test set to predict classification accuracy
y_pred = predict(decTree, newdata = test_set[-7], type = 'class')

# Plot the tree
plot(decTree)

# Evaluate fitting using model summary
# View model's summary
summary(decTree)

# Evaluate model prediction error using confusion matrix
cm = as.matrix(table(test_set[, 7], y_pred))
library(gmodels)
gmodels::CrossTable(test_set$car,
           y_pred,
           prop.chisq = FALSE,
           prop.c     = FALSE,
           prop.r     = FALSE,
           dnn = c('actual default', 'predicted default'))

# Evaluate model prediction error using precision-recall and F1 score
precision = diag(cm) / colSums(cm)
recall = diag(cm) / rowSums(cm)
f1_score = ifelse(precision + recall == 0, 0, 2 * precision * recall / (precision + recall))

# Evaluate splitting rule using learning curve
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

# Predict user decision using model
decision = list(persons=4, buying='med', maint='low',
                lug_boots='med', safety='med', doors=4)
# Since doors and persons are categorial, then we need transform it into factor
decision$doors = factor(decision$doors, levels = c(2, 3, 4, '5more'))
decision$persons = factor(decision$persons, levels = c(2, 4, 'more'))
pred_dec = predict(decTree, newdata = decision, type = 'class')
pred_dec
