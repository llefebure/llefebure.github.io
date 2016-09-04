#' Read in data for analysis
#'
#' @description Connect to the raw sqlite database, filter for desired data,
#' and do light processing to prepare for analysis.
#' @param dir, the path of the data directory where the database lives
#' @param db, the name of the database filename
#' @param fn, the name of the rds data file (if exists, otherwise it will be created in the
#' current working directory)
#' @return a data frame with the processed data
readData <- function(dir = "../data/", db = "database.sqlite", fn = "data.rds") {
  if (file.exists(fn)) {
    df <- readRDS(fn)
  } else {
    df <- src_sqlite(paste0(dir, db), create = F) %>% 
      tbl("May2015") %>%
      filter(subreddit == "soccer") %>% # look at only the soccer subreddit
      collect()
    df <- df %>% 
      mutate(created_timestamp = as.POSIXct(created_utc, tz = "UTC", origin = "1970-01-01"),
             created_date = as.Date(format(created_timestamp, "%Y-%m-%d")),
             body_length = length(str_split(body, " ")[[1]]))
    saveRDS(df, fn)
  }
  return(df)
}