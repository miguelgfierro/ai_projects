library(quanteda)
library(readr)


# Trim sentence to max lenght, remove odd characters and put to lower
str.manipulate <- function(x, max.text.lenght=500){
  x <- substr(x, 1, max.text.lenght)
  Encoding(x) <- "UTF-8"
  x <- toLower(x)
  return(x)
}


map.text.to.number <- function(data, alphabet, nomatch = 0){
  alphabet.list <- strsplit(alphabet, split = "")[[1]]
  xx <- strsplit(data, split="")
  xx <- lapply(xx, rev)
  data.map <- lapply(xx, match, table=alphabet.list, nomatch = nomatch)
  data.map
}

text.encoder.csv <- function(input.file, output.file, alphabet, max.text.lenght,
                             shuffle=FALSE, verbose=TRUE, nomatch=0){
  #read file
  if(verbose) cat("Read input file\n")
  dt <- read_csv(input.file, 
                 col_names = c('V1','V2', 'V3'),
                 col_types = "icc",
                 n_max = Inf
  )
  
  #split label and data
  ysplit <- dt$V1
  dt <- with(dt, paste(V2, V3, sep=""))
  dt <- str.manipulate(dt, max.text.lenght)
  
  #map text to number
  if(verbose) cat("Compute the dictionary\n")
  data.map <- map.text.to.number(dt, alphabet, nomatch)
  
  #transform to df
  data.map.df <- as.data.frame(do.call(rbind,lapply(data.map, `length<-`, max.text.lenght)))
  data.map.df[is.na(data.map.df)] <- nomatch
  
  #add label
  cols <- colnames(data.map.df)
  data.map.df$label <- ysplit
  new.cols <- c("label", cols)
  data.map.df <- data.map.df[,new.cols]
  
  #shuffle
  if (shuffle){
    if(verbose) cat("Shuffle data\n")
    data.map.df <- data.map.df[sample(nrow(data.map.df)),]
  }
  
  #write to ouput
  if(verbose) cat("Write output file\n")
  write.table(data.map.df, output.file, col.names=FALSE, row.names=FALSE, sep=",")
}

dict.decoder <- function(data, alphabet, feature.len, batch.size){
  vocab.size <- nchar(alphabet)
  feature.matrix <- apply(data, 2, function(x){
    im <- matrix(0L, feature.len, vocab.size)
    for(i in seq_along(x)){
      im[i,x[i]] <- 1L
    }
    im
  })
  dim(feature.matrix) <- c(feature.len, vocab.size, 1,  batch.size)
  feature.matrix
}


