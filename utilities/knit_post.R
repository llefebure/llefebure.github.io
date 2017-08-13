# adapted from Andrew Brooks
# (http://brooksandrew.github.io/simpleblog/articles/blogging-with-r-markdown-and-jekyll-using-knitr/)

# Dump Rmd file into _R/<file_name>.Rmd and run KnitPost(<file_name>.Rmd)

KnitPost <- function(rmd.name) {
  
  site.path <- '~/Documents/Projects/llefebure.github.io/'
  
  library(knitr)
  library(stringr)
  
  ## Blog-specific directories. This will depend on how you organize your blog.
  rmd.path <- paste0(site.path, "_R/", rmd.name) # file path for Rmd to knit
  fig.dir <- paste0("assets/Rfig/") # directory to save figures
  posts.path <- paste0(site.path, "_posts/") # directory for converted markdown files
  cache.path <- paste0(site.path, "_cache") # necessary for plots
  
  render_jekyll(highlight = "pygments")
  opts_knit$set(base.url = '/', base.dir = site.path)
  opts_chunk$set(fig.path=fig.dir, fig.width=8.5, fig.height=5.25, cache=F, 
                 warning=F, message=F, cache.path=cache.path, dev = "svg", tidy=F)   
  
  # render markdown
  out.file <- knit(rmd.path,
                   output = paste0(posts.path, str_replace(rmd.name, ".Rmd", ".md")),
                   envir = parent.frame(), 
                   quiet = T)
}
