ListArr2List = function(arr){
  n = row.names(arr);
  l = lapply(arr,unlist);
  names(l) = n;
  return(l);
}