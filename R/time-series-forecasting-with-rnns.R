### Time Series Forecasting with Recurrent Neural Networks ----
### (available online: https://blogs.rstudio.com/ai/posts/2017-12-20-time-series-forecasting-with-recurrent-neural-networks/)

library(data.table)
library(lattice)
library(keras)


### sample data ----

csv = "inst/extdata/jena_climate_2009_2016.csv"

## if applicable, download and extract .csv file
if (!file.exists(csv)) {
  
  zip = paste0(
    basename(csv)
    , ".zip"
  )
  
  dst = file.path(
    tempdir()
    , zip
  )
  
  utils::download.file(
    file.path(
      "https://s3.amazonaws.com/keras-datasets"
      , zip
    )
    , destfile = dst
  )
  
  utils::unzip(
    dst
    , files = basename(csv)
    , exdir = dirname(csv)
  )
}

## read data
dat = fread(csv)

## convert time column to `POSIXct`
dat[
  , `Date Time` := as.POSIXct(
    `Date Time`
    , tz = "UTC"
    , format = "%d.%m.%Y %H:%M:%S"
  )
]

## vis
dat[
  , xyplot(
    `T (degC)` ~ `Date Time`
    , panel = \(x, y) {
      panel.abline(h = 0L, col = "grey", lty = 2)
      panel.xyplot(x, y, type = "l")
    }
  )
]


### data prep ----

#### scaling ----

mat = dat[
  , data.matrix(.SD)
  , .SDcols = is.numeric
]

trn = mat[1:2e5, ]
avg = apply(trn, MARGIN = 2L, FUN = mean)
std = apply(trn, MARGIN = 2L, FUN = sd)
scl = scale(mat, center = avg, scale = std)


#### data generator ----

# @param data original array of floating-point data
# @param lookback how many timesteps back the input data should go
# @param delay how many timesteps in the future the target should be
# @param min_index,max_index indexes delimiting which timesteps to draw from
# @param shuffle whether to shuffle the samples or draw in chronological order
# @param batch_size number of samples per batch
# @param step the period at which to sample data
generator = function(
    data
    , lookback
    , delay
    , min_index
    , max_index = NULL
    , shuffle = FALSE
    , batch_size = 128L
    , step = 6L
) {
  
  # # debug:
  # data = scl
  # lookback = 10 * 24 * 6
  # delay = 24 * 6
  # min_index = 1L
  # max_index = 2e5L
  # shuffle = TRUE
  # batch_size = 128L
  # step = 6L
  
  if (is.null(max_index)) {
    max_index = nrow(data) - delay - 1L
  }
  
  i = min_index + lookback
  
  function() {
    
    if (shuffle) {
      rows = sample(
        (min_index+lookback):max_index
        , size = batch_size
      )
    } else {
      if (i + batch_size >= max_index) {
        i <<- min_index + lookback
      }
      
      rows = i:min(
        i + batch_size - 1L
        , max_index
      )
      
      i <<- i + length(rows)
    }
    
    # sample -> one batch of input data
    samples = array(
      0
      , dim = c(
        length(rows)
        , lookback / step
        , dim(data)[[-1]]
      )
    )
    
    # target -> corresponding array of target temperatures
    targets = array(
      0
      , dim = length(rows)
    )
    
    for (j in 1:length(rows)) {
      indices = seq(
        rows[[j]] - lookback
        , rows[[j]] - 1L
        , length.out = dim(samples)[[2]]
      )
      
      samples[j,,] = data[indices, ]
      targets[[j]] = data[rows[[j]] + delay, "T (degC)"]
    }
    
    list(
      samples
      , targets
    )
  }
}

lookback = 1440L # 10 days worth of 10-minute readings
step = 6L # observations sampled at one data point per hour
delay = 144L # 24 hours worth of 10-minute readings
batch_size = 128L

## training generator: looks at first 200,000 timesteps
train_gen = generator(
  scl
  , lookback = lookback
  , delay = delay
  , min_index = 1L
  , max_index = 200000L
  , shuffle = TRUE
  , step = step
  , batch_size = batch_size
)

## validation generator: looks at following 100,000 timesteps
val_gen = generator(
  data
  , lookback = lookback
  , delay = delay
  , min_index = 200001
  , max_index = 300000
  , step = step
  , batch_size = batch_size
)

## training generator: looks at the remainder
test_gen = generator(
  data
  , lookback = lookback
  , delay = delay
  , min_index = 300001
  , step = step
  , batch_size = batch_size
)

# How many steps to draw from val_gen in order to see the entire validation set
val_steps = (300000 - 200001 - lookback) / batch_size

# How many steps to draw from test_gen in order to see the entire test set
test_steps = (nrow(data) - 300001 - lookback) / batch_size


### baseline model ----

## "[A] a common-sense approach is to always predict that the temperature 24 
## hours from now will be equal to the temperature right now."

evaluate_naive_method = function() {
  batch_maes = c()
  batch_rmses = c()
  for (step in 1:val_steps) {
    c(samples, targets) %<-% val_gen()
    preds = samples[, dim(samples)[[2]], 2] # i.e. "T (degC)"
    mae = mean(abs(preds - targets))
    batch_maes = c(batch_maes, mae)
    batch_rmses = c(batch_rmses, mltools::rmse(preds, targets))
  }
  lapply(
    list(batch_maes, batch_rmses)
    , mean
  )
}

c(mae, rmse) %<-% evaluate_naive_method()

mae * std[[2]]
rmse * std[[2]]


### basic ml approach ----

model = keras_model_sequential() %>% 
  layer_flatten(input_shape = c(lookback / step, dim(scl)[-1])) %>% 
  layer_dense(units = 32, activation = "relu") %>% 
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history = model %>% fit(
  train_gen
  , steps_per_epoch = 500
  , epochs = 20
  , validation_data = val_gen
  , validation_steps = val_steps
)


### recurrent baseline ----

model = keras_model_sequential() %>% 
  layer_gru(units = 32, input_shape = list(NULL, dim(scl)[[-1]])) %>% 
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history = model %>% fit(
  train_gen
  , steps_per_epoch = 500
  , epochs = 20
  , validation_data = val_gen
  , validation_steps = val_steps
)
