df=read.csv("new_train.csv");head(df)
df[,c(2,3,4,5,6,7,8,9,10,15,16)]=lapply(df[,c(2,3,4,5,6,7,8,9,10,15,16)], FUN = as.factor)
as.factor(df[,c(2,3,4,5,6,7,8,9,10,15,16)])
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
library(rpart)
library(rpart.plot)
####NOW PARTITIONG THE TRAINING DATASET INTO VALIDATION PART BECAUSE THE GIVEN DATASET IS MUCH LARGER
partidx=sample(1:nrow(df1),26000,replace = F)#20% of train dataset convert to validation dataset
df1train=df1[partidx,];head(df1train)#training
partidx1=sample(1:nrow(df1[-partidx]),6950,replace=F)
intersect(partidx,partidx1)
df1valid=df1[partidx1,]
head(df1valid)#VALIDATION
#BUILD MODEL ON TRAINING PARTITION
mod1=rpart(y~.,method = "class",data=df1train,control = rpart.control(
  cp=0,minsplit = 2,minbucket = 1,maxcompete = 0,maxsurrogate = 0,xval = 0,
  parms=list(split="gini")))
predicted_y=predict(mod1,dftest,type = "class")
df1test=cbind(dftest,predicted_y);head(df1test)
##########PRUNING PROCESS##################

#AVOID OVERFITTING
#* FULL GROWN TREE LEADS TO COMPLETE OVERFITTING OF DATA
#* POOR PERFORMANCE ON NEW DATA
#* PRUNE THE FULL GROEN TREE BACK TO A LEVEL WHERE IT DOESN'T OVERFIT THE DATA OR FIT NOISE

#PRUNING PROCESS
#VALIDATION PARTITION: MISCLASSIFICATION ERROR VS NUMBER OF DECISION NODES
#TOTAL NUMBER OF NODES IN FULL GROWN TREES
nrow(mod1$frame)
#number of decision nodes
nrow(mod1$splits)
#no of terminal nodes
nrow(mod1$frame)-nrow(mod1$splits)
#NODE NUMBER
head(row.names(mod1$frame))
#COERCION TO INTEGER
tosses1=as.integer(row.names(mod1$frame))
tosses2=sort(tosses1);head(tosses2)
#COUNTER FOR NODES TO BE SNIPPED OFF
i=1
mod1splitv=NULL
mod1strainv=NULL
mod1svalidv=NULL
errtrainv=NULL
errvalidv=NULL
for (y in mod1$frame$var) {
  if (as.character(y)!="<leaf>" & i<length(tosses2)) {
    tosses3=tosses2[(i+1):length(tosses2)]
    mod1split=snip.rpart(mod1,toss = tosses3)
    mod1splitv=c(mod1splitv,mod1split)
    mod1strain=predict(mod1split,df1train,type="class")
    mod1strainv=c(mod1strainv,mod1strain)
    #for validation dataset 
    mod1svalid=predict(mod1split,df1valid,type="class")
    mod1svalidv=c(mod1svalidv,mod1svalid)
    errtrain=mean(mod1strain!=df1train$y)
    errtrainv=c(errtrainv,errtrain)
    errvalid=mean(mod1svalid!=df1valid$y)
    errvalidv=c(errvalidv,errvalid)
  }
  i=i+1
}

#ERROR RATE VS NO OF SPLITS
DF=data.frame("DECISION_NODES"=1:nrow(mod1$splits),"ERROR_TRAINING"=errtrainv,
              "ERROR_VALIDATION"=errvalidv,check.names=F)

DF#HERE WE SEE RATE OF DECREASE IN THE TRAINING PARTITION OR VALIDATION PARTITION
##TREE AFTER LAST SNIP
prp(mod1split,varlen = 0,cex = 0.7,extra = 0,compress = T,Margin = 0,digits = 0)
nrow(mod1split$frame)
nrow(mod1split$splits)
nrow(mod1split$frame)-nrow(mod1split$splits)
#plot of error rate VS number of splits

nsplits=1:nrow(mod1$split)
plot(nsplits,100*errtrainv,type = "l",xlab = "NUMBER OF SPLITS",ylab = "ERROR_RATE")
lines(nsplits,100*errvalidv,col="red")
#MINIMUM ERROR TREE AND BEST PRUNED TREEE
min(errvalidv)
MET=min(nsplits[which(errvalidv==min(errvalidv))]);MET
#STANDARD ERROR
sqrt(var(errvalidv)/length(errvalidv))
#####################
#BEST PRUNED TREE NEAR FIRST MINIMA WITHIN STANDARD ERROR
met1std=min(errvalidv)+sqrt(var(errvalidv)/length(errvalidv));met1std
BPT=DF[which(errvalidv > min(errvalidv) & errvalidv < met1std&nsplits,MET),][1,1]
BPT#here the resulting decision node are required so we can removing remaining node after this resulting node

#BUT THIS IS NOT EXACT ANSWER 2-3 NODES GIVES BETTER RESULT THAN THIS ABOVE  RESULTING NODE IN BPT

toss3=tosses2[(BPT+1):length(tosses2)]
mod1best=snip.rpart(mod1,toss = toss3)
prp(mod1best,varlen = 0,cex=0.7,extra=0,compress = T,Margin = 0,digits = 0,split.cex = 0.8,under.cex = 0.8)
#SIMILARLY WE FIND CLASSIFICATION ACCURACY & MISCLASSIFICATION ERROR FOR TRAINING, VALIDATION & TESTING DATASET
bmodtrain1=predict(mod1best,df1train,type="class")
table("PREDICTED_VALUE"=bmodtrain1,"ACTUAL_VALUE"=df1train$y)
#CLASSIFICATION ACCURACY
mean(bmodtrain1==df1train$y)#96% ACCURATE MODEL
#CLASSIFICATION ERROR
mean(bmodtrain1!=df1train$y)

########  in validation partition#######
bmodvalid1=predict(mod1best,df1valid,type="class")
table("ACTUAL_VALUE"=df1valid$y,"PREDICTED_VALUE"=bmodvalid1)

#CLASSIFICATION ACCURACY
mean(bmodvalid1==df1valid$y)#*SO HERE WE ALSO SEE THAT ACCURACY IS 97% AND ERROR PERFORMANCE BIT STABLE
                            #*AS COMPARE TO THE TRAINING DATASET.
#CLASSIFICATION ERROR
mean(bmodvalid1!=df1valid$y)
##########in testing partition########
bmodtest1_y=predict(mod1best,df1test,type="class")
table("ACTUAL_VALUE"=df1test$predicted_y,"PREDICTED_VALUE"=bmodtest1_y)
df1test=cbind(df1test,bmodtest1_y)
head(df1test)
write.csv(df1test,file = "BEST_PREDICTED_Y.csv")

#CLASSIFICATION ACCURACY
mean(bmodtest1==df1test$predicted_y)#*ON TESTING DATASET ACCURACY IS ALSO 97% WHICH IS QUITE  BETTER MODEL
#CLASSIFICATION ERROR
mean(bmodtest1!=df1test$predicted_y)
