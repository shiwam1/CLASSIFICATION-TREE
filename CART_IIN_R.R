list.files()
df=read.csv("new_train.csv");head(df)
df[,c(2,3,4,5,6,7,8,9,10,15,16)]=lapply(df[,c(2,3,4,5,6,7,8,9,10,15,16)], FUN = as.factor)
str(df)
levels(df$job)=0:(length(levels(df$job))-1)
levels(df$marital)=0:(length(levels(df$marital))-1)
levels(df$contact)=0:(length(levels(df$contact))-1)
levels(df$education)=0:(length(levels(df$education))-1)
levels(df$default)=0:(length(levels(df$default))-1)
levels(df$housing)=0:(length(levels(df$housing))-1)

levels(df$loan)=0:(length(levels(df$loan))-1)
levels(df$month)=0:(length(levels(df$month))-1)
levels(df$day_of_week)=0:(length(levels(df$day_of_week))-1)
levels(df$poutcome)=0:(length(levels(df$poutcome))-1)
levels(df$y)=0:(length(levels(df$y))-1)
df[,c(2,3,4,5,6,7,8,9,10,15,16)]=lapply(df[,c(2,3,4,5,6,7,8,9,10,15,16)], FUN = as.character)
df[,c(2,3,4,5,6,7,8,9,10,15,16)]=lapply(df[,c(2,3,4,5,6,7,8,9,10,15,16)], FUN = as.numeric)
head(df)#NOW ALL ARE IN NUMERIC VARIABLE
df1=df[,-c(13,14)]
dftest=read.csv("new_test.csv");head(dftest)
# VISUALIZING THIS DATASET USING MULTIVARIATE PLOTS
#######PARALLEL COORDINATE PLOTS#############
palette("default")
library(MASS)
parcoord(df1[which(df1$y=="0"),],col = "blue")
axis(2,at=axTicks(2),labels = c("0%","20%","30%","40%","50%","60%"))
grid()
parcoord(df1[which(df1$y=="1"),],col = "red")
axis(2,at=axTicks(2),labels = c("0%","20%","30%","40%","50%","60%"))
grid()
##APPLYING CART MODEL##
##USING LIBRARY(RPART)
library(rpart)
library(rpart.plot)

mod=rpart(df1$y~.,method = "class",data = df1,
          control = rpart.control(cp=0,minsplit = 2,minbucket = 1,
          maxcompete = 0,maxsurrogate = 0,xval = 0,parms=list(split="gini")))

##node number is also include in it if we add nn and nn.cex in prp command
#prp(mod,type = 1,extra = 1,under = T,varlen = 0,cex=0.7,compress = T,Margin = 0,digits = 0,split.cex = 0.8,under.cex = 0.8,nn=T,nn.cex = 0.6)



####NOW PARTITIONG THE TRAINING DATASET INTO VALIDATION PART BECAUSE THE GIVEN DATASET IS MUCH LARGER
partidx=sample(1:nrow(df1),26000,replace = F)#20% of train dataset convert to validation dataset
df1train=df1[partidx,];head(df1train)#training
partidx1=sample(1:nrow(df1[-partidx]),6950,replace=F)
#intersect(partidx,partidx1)
df1valid=df1[partidx1,]
head(df1valid)#VALIDATION
#BUILD MODEL ON TRAINING PARTITION
mod1=rpart(y~.,method = "class",data=df1train,control = rpart.control(
  cp=0,minsplit = 2,minbucket = 1,maxcompete = 0,maxsurrogate = 0,xval = 0,
  parms=list(split="gini")))
par(mar=c(0,0,0,0),oma=c(0,0,0,0),xpd=NA)
plot(mod1,uniform = T,branch = 0.1,compress = T,margin = 0,nspace = 1)
text(mod1,splits = T,use.n = F,all=F,minlength = 0,cex=0.7)#here all the text shon in full grown tree


#####NICER VERSION OF THIS FULL GROWN TREE###########
#prp(mod1,varlen = 0,cex=0.7,extra = 0,compress = T,Margin = 0,digits = 0,nn=T,nn.cex=0.6)
#FIRST FOUR LEVEL OF FULL GROWN TREE
toss1=as.integer(row.names(mod1$frame))
toss2=sort(toss1)
toss3=toss2[which(toss2==16):length(toss2)]
mod1sub=snip.rpart(mod1,toss=toss3)
prp(mod1sub,varlen = 0,cex=0.7,extra=0,compress = T,Margin = 0,digits = 0)#ALL THE NODES SHOWS CLEARLY EXCEPT 16 LAST NODE
#DESCRIPTION OF EACH SPLITTING STEP OF THE FULL GROWN TREE

#TOTAL NUMBER OF NODES
nrow(mod1$splits)
#NUMBER OF TOTAL NODES
nrow(mod1$frame)
#NUMBER OF TERMINAL NODES
nrow(mod1$frame)-nrow(mod1$splits)# WHICH IS ONE MORE THAN DECISION NODES
#THIS IS THE PROPERTY OF BINARY TREES, THE NUMBER OF TERMINAL NODES OR NUMBER OF 
#LEAFS ARE ONE MORE THAN THE DECISION NODES
################################################################################

#IF WE ARE INTERESTED IN HAVING A TABLE WHERE WE HAVE THE INFORMATION ABOUT THE VARIABLES
#WHICH HAVE BEEN USED FOR SPLITTING AND THE SPLIT VALUE, THE PREDICTOR VALUE COMBINATION THEN
#THIS IS DONE AS FOLLOWS:-


#j is the counter of split variables
#i is the counter of split values
splitvalue=NULL
j=1
i=1
for (x in mod1$frame$var) {
  if(as.character(x)!="<leaf>"){
    if (!is.factor(df1[,as.character(x)])) {
      splitvalue[i]=mod1$splits[j,"index"]
    }
    else{
      cl=NULL
      #split variable is factor
      #k={1,largest number of levels in the factor}
      
      for (k in 1:ncol(mod1$csplit)) {
        temp=mod1$csplit[mod1$splits[j,"index"],k]
        #if level(temp)goes to the left child
        if (temp==1) {
          cl=paste(cl,levels(df1[,as.character(x)])[k],sep = ",")
        }
      }
      splitvalue[i]=substr(cl,start = 2,stop = nchar(cl))
    }
    j=j+1
  }
  else{
    splitvalue[i]=NA
  }
  i=i+1
}
dat=data.frame("NODE_NUMBER"=row.names(mod1$frame),
           "SPLIT_VAR"=mod1$frame$var,
           "SPLIT_VALUE"=splitvalue,
           "CASES"=mod1$frame$n,
           "CLASS"=mod1$frame$yval-1,check.names = F)
head(dat,20)
#Prediction For Training Dataset
mod1train=predict(mod1,df1train,type = "class")
table("ACTUAL_VALUE"=df1train$y,"PREDICTED_VALUE"=mod1train)
#CLASSIFICATION ACCURACY
mean(mod1train==df1train$y)
#misclassification error
mean(mod1train!=df1train$y)
#PREDICTION FOR VALIDATION DATASET
mod1valid=predict(mod1,df1valid,type = "class")
table("ACTUAL_VALUE"=df1valid$y,"PREDICTED_VALUE"=mod1valid)#some error are seen here
#CLASSIFICATION ACCURACY
mean(mod1valid==df1valid$y)
#misclassification error
mean(mod1valid!=df1valid$y)
######################################3
#PREDICTION TO TESTING DATASET
predicted_y=predict(mod1,dftest,type = "class")
df1test=cbind(dftest,predicted_y);head(df1test)
#######################################

#ABOVE MODEL IS OVERFITTING CASE BECAUSE ACCURACY OF TRAING DATASET IS 1 AND ERROR IS 0.
#SO IT WILL FIT THE NOISE OF THE DATA

















































































