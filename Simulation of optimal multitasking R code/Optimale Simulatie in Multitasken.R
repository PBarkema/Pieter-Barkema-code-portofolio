# Authors: Matthijs Wolters & Pieter Barkema
# Universiteit Utrecht

# Working directory
setwd("C:\\Users\\piete\\OneDrive\\Documenten\\Exp_Methode_Statistiek")
load("allData.Rdata")
#load("MonteCarloFrame.Rdata")
require(gdata)

##
##clean data set
##
cleanData <- allData[allData$LocalTime <= 150,]

###
###analysis of wordscore 14
###
ourData14 <- cleanData[cleanData$WordScore == 14 & cleanData$partOfExperiment == "dualTask"&cleanData$LocalTime <= 150,]
ourDataDrop14 <- drop.levels(ourData14)

###
###analysis of wordscore 28
###
ourData28 <- cleanData[cleanData$WordScore == 28&cleanData$partOfExperiment == "dualTask"&cleanData$LocalTime <= 150,]

###
###analysis of easy scrabble
###
ourDataEasy <- cleanData[cleanData$partOfExperiment == "dualTask"&cleanData$LocalTime <= 150&cleanData$scrabbleCondition == "easy",]

###
###analysis of hard scrabble
###
ourDataHard <- cleanData[cleanData$partOfExperiment == "dualTask"&cleanData$LocalTime <= 150&cleanData$scrabbleCondition == "hard",]


###
###finding a regression for the practice type task
###
###
practiceTypeData <- cleanData[cleanData$partOfExperiment == "practiceTyping" & cleanData$scrabbleTaskPresent == "False",]
allSubjectData <- data.frame(Time = practiceTypeData$LocalTime, Letters = practiceTypeData$LengthOfUserLetterString)
allSubjectData2 <- drop.levels(allSubjectData)
allSubjectData2[order(allSubjectData2$Time),]
typeRegr <- summary(lm(allSubjectData2$Letters~allSubjectData2$Time))$coefficients



importantVals <- data.frame(typeInt = typeRegr[1,1], typeIntSD = typeRegr[1,2], typeSlope = typeRegr[2,1], typeSlopeSD = typeRegr[2,2])




###
###practice scrabble analysis
###thrown out data points where the lettertask was also present
###vervolg onderzoek moet laten zien of deze regressie wel de data correct representeert
practiceScrabbleData <- cleanData[cleanData$partOfExperiment == "practiceScrabble"&cleanData$letterTaskPresent == "False"&cleanData$Eventmessage1 == "keypressScrabble"&cleanData$CurrentScore>0&cleanData$LocalTime>0,]
practiceScrabbleData2 <- drop.levels(practiceScrabbleData)

practiceScrabbleData2$y <- (practiceScrabbleData2$CurrentScore/practiceScrabbleData2$WordScore)
practiceScrabbleData2$x <- practiceScrabbleData2$LocalTime

scrabbleRegr<-summary(nls(y~b*x^(a), data=practiceScrabbleData2, start=list(a=0.5, b=1)))$coefficients
#data.frame(typeInt = typeRegr[1,1], typeIntSD = typeRegr[1,2], typeSlope = typeRegr[2,1], typeSlopeSD = typeRegr[2,2])
importantVals$scrabbleExp <- scrabbleRegr[1,1]
importantVals$scrabbleExpSD <- scrabbleRegr[1,2]



dimReturn<-summary(nls(y~x*a^(x)+b, data=practiceScrabbleData2, start=list(a=0.6, b=0)))$coefficients
#testing words per second instead of score per second
wordRegr <- summary(lm((practiceScrabbleData2$CurrentScore/practiceScrabbleData2$WordScore)~practiceScrabbleData2$LocalTime))$coefficients
importantVals$scrabbleFac <- scrabbleRegr[2,1]
importantVals$scrabbleFacSD <- scrabbleRegr[2,2]
importantVals$scrabbleInt <- wordRegr[1,1]
importantVals$scrabbleIntSD <- wordRegr[1,2]

#
#
#checking the non linear regressions of the scrabble data
#
#
eqDim <- function(x){}

eqRoot <- function(x){0.641*x^(0.592)}
eqRootEasy<- function(x){0.72*x^(0.61)}
eqRootHard <- function(x){0.59*x^(0.57)}

plot(eqRoot(1:150), col="purple", ylim=c(0,30))
points(eqRootEasy(1:150), col="red")
points(eqRootHard(1:150), col="blue")
axis(side=2,at=seq(0,600,25),labels=rep("",length(seq(0,30,5))))
points(practiceScrabbleData2$x, practiceScrabbleData2$y)

eq <- function(x){x*0.9667516^x}
eq2 <- function(x){9.541*x^0.431}
plot(MonteCarloFrame[MonteCarloFrame$trial==1,]$time, MonteCarloFrame[MonteCarloFrame$trial==1,]$currentScore, col="red", ylim=c(0,600))
#points(eq2(1:150), col="red")
axis(side=2,at=seq(0,600,25),labels=rep("",length(seq(0,600,25))))
points(MonteCarloFrame[MonteCarloFrame$trial==2,]$time, MonteCarloFrame[MonteCarloFrame$trial==2,]$currentScore)


#
#defining a dataframe to store all our generated data in
#
generatedData <- data.frame(subject=0, trial=0, wordScore=14, task="Scrabble", time=0, currentScore=0, logMessage="Generate dataframe")

#
#creating a function to evaluate the points gained in 
#
evalPoints <- function(data=generatedData, iter=2){#1==Scrabble&2==Type
  dataF <- tail(data, n=1)
  iterLength <- seq(iter)
  if(dataF$task=="Scrabble"){
    fctr <- importantVals$scrabbleFac
    stdDevF <- importantVals$scrabbleFacSD
    expn <- importantVals$scrabbleExp
    stdDevE <- importantVals$scrabbleExpSD
    
    for (t in iterLength){
      val <- 1
      updateData <- tail(dataF,n=1)
      fctrValue <- rnorm(1, fctr, stdDevF)
      expValue <- rnorm(1, expn, stdDevE)
      #slope <- (genValue^updateData$time * (1 + updateData$time * log(genValue)))
      updateData$time <- updateData$time + 1
      wordFound <- fctrValue*(updateData$time)^expValue
      if(wordFound*updateData$wordScore>updateData$currentScore+updateData$wordScore){
        updateData$currentScore <- updateData$currentScore + updateData$wordScore
        updateData$logMessage <- "correct newWord"
        rownames(updateData) <- nrow(data)+1
        dataF <- rbind(dataF, updateData)
      }
      # if(genValue==0){
      #   rownames(updateData) <- nrow(data)+1
      #   updateData$logMessage <- "No Word Found"
      #   dataF <- rbind(dataF, updateData)
      # }
    }
  }
  else if(dataF$task=="Type"){#currentTask==2
    slope <- importantVals$typeSlope
    stdDev <- importantVals$typeSlopeSD
    #genValue <- round(rnorm(time, slope, stdDev))
    
    for (t in iterLength){
        val <- 1
        updateData <- tail(dataF, n=1)
        updateData$time <- updateData$time + 1
        genValue <- round(rnorm(1, slope, stdDev))
        rowname<-nrow(data)+nrow(dataF)-1
        
        while(val<=genValue){
          updateData$currentScore <- updateData$currentScore +1
          updateData$logMessage <- "correct keypress"
          rownames(updateData) <- rowname+val
          dataF <- rbind(dataF, updateData)
          val <- val +1
        }
        if(genValue==0){
          rownames(updateData) <- length(dataF$trial)+1
          updateData$logMessage <- "No Letter Typed"
          dataF <- rbind(dataF, updateData)
        }
    }
    
  }
  data <- rbind(data, dataF[-(1),])
  #data
  #print(data)
}

#
#creating a function to evaluate whether to switch tasks or not
#
swapTask <- function(data, switchCost=10){#1==Scrabble&2==Type
  dataF <- tail(data, n=1)
  
  slope<-rnorm(1,importantVals$typeSlope,importantVals$typeSlopeSD)
  intersect<-rnorm(1,importantVals$typeInt, importantVals$typeIntSD)
  exponent <- rnorm(1, importantVals$scrabbleExp,importantVals$scrabbleExpSD)
  factorB <- rnorm(1, importantVals$scrabbleFac,importantVals$scrabbleFacSD)
  #intersectB <- rnorm(1, importantVals$scrabbleInt, importantVals$scrabbleIntSD)
  
  
  if(dataF$task=="Type"){
    switchType<-0
    switchScrabble<-switchCost
    
  }
  else if(dataF$task=="Scrabble"){
    switchType<-switchCost
    switchScrabble<-0
  }
  
  estProfitType <- (slope*(dataF$time+2+switchType)+intersect)#the 2 represents the iteration length of 2 seconds
  estProfitScrabble <- (factorB*(dataF$time+2+switchScrabble)^exponent)*dataF$wordScore #(factorB*(dataF$time+2+switchScrabble)+intersectB)
  if(estProfitScrabble-estProfitType>0&dataF$task=="Type"){switchGuess<-T}
  else if(estProfitType-estProfitScrabble>0&dataF$task=="Scrabble"){switchGuess<-T}
  else{switchGuess<-F}
  switchGuess
}



singleDataSet<-function(data, pop=1, trialAmnt=2,time=150) {
  pop<-seq(pop)
  trial<-seq(trialAmnt)

  
  for(p in pop){ 
    for(t in trial){
      startTask <- sample(c("Scrabble", "Type"), 1)
      #print(startTask)
      trialData <- data.frame(subject=p, trial=t, wordScore=tail(data$wordScore, n=1), task=startTask, condition=tail(data$condition,n=1), time=0, currentScore=0, logMessage=paste("starting", startTask,"task", sep=" "))#, row.names = t)
      trialTime <- 1
      
      while(trialTime<time){
        #print("running point eval")
        trialData <- evalPoints(trialData)
        #print("running swap task")
        switchGuess <- swapTask(trialData)
        if (switchGuess){
          #print("swap is true")
          tempData <- tail(trialData, n=1)
          if(tempData$task=="Scrabble"){tempData$task<-"Type"}
          else{tempData$task<-"Scrabble"}
          tempData$logMessage <- "Swapping Task"
          rownames(tempData)<-nrow(trialData)+1
          trialData <- rbind(trialData, tempData)
        }
        trialTime<-tail(trialData$time, n=1)
        #print(trialTime)
      }
      data<-rbind(data, trialData)

    }
  }
  frame<-data

  return(frame)
}

fullModel <- function(wordS="None", cond="None"){
  genData <- data.frame(subject=0, trial=0, wordScore=wordS, task="None", condition="None", time=0, currentScore=0, logMessage="Generate dataframe")
  MonteCarloFrame <- data.frame(subject=0, trial=0, wordScore=0, task="None", condition="None", time=0, currentScore=0, logMessage="FullDataSet")
  if(wordS=="None"){
    if (cond=="hard"){
      name<-cond
      importantVals$scrabbleExp <- 0.57
      importantVals$scrabbleFac <- 0.59
      for(w in c(14,28)){
        genData$wordScore <- w
        genData$condition <- cond
        singleDS <- singleDataSet(genData, 155, 4)
        MonteCarloFrame <- rbind(MonteCarloFrame, singleDS)
      }
    }
    else if (cond=="easy"){
      importantVals$scrabbleExp <- 0.61
      importantVals$scrabbleFac <- 0.72
      name<-cond
      for(w in c(14,28)){
        genData$wordScore <- w
        genData$condition <- cond
        singleDS <- singleDataSet(genData, 155, 4)
        MonteCarloFrame <- rbind(MonteCarloFrame, singleDS)
      }
    }
  }
  
  if(cond=="None"){
    if (wordS==14){
      name<-wordS
      genData$wordScore <- wordS
      for(c in c("easy", "hard")){
        genData$condition <- c
        if(c=="easy"){
          importantVals$scrabbleExp <- 0.61
          importantVals$scrabbleFac <- 0.72}
        else if (c=="hard"){
          importantVals$scrabbleExp <- 0.57
          importantVals$scrabbleFac <- 0.59}
        singleDS <- singleDataSet(genData, 155, 4)
        MonteCarloFrame <- rbind(MonteCarloFrame, singleDS)
      }
    }
    else if (wordS==28){
      name<-wordS
      genData$wordScore <- wordS
      for(c in c("easy", "hard")){
        genData$condition <- c
        if(c=="easy"){importantVals$scrabbleExp <- 0.61
        importantVals$scrabbleFac <- 0.72}
        else if (c=="hard"){importantVals$scrabbleExp <- 0.57
        importantVals$scrabbleFac <- 0.59}
        singleDS <- singleDataSet(genData, 155, 4)
        MonteCarloFrame <- rbind(MonteCarloFrame, singleDS)
      }
    }
  }
  save(MonteCarloFrame, file=paste("MonteCarloFrame", name,".RData", sep=" "))
}


# paired t-test
# y1: Van onze data, per proefpersoon een gemiddelde, over alle EINDscores van alle de 4 dualtask trials
# y2: van MonteCarlo data, per proefpersoon een gemiddelde, over alle EINDscores van de 4 dualtask trials
# 1085 trials zouden er moeten zijn: 7*155. Dat klopt(?): 1084 * trialStopTooMuchTime; 1* trialStop
y2 <- data.frame(mean=0,subject=0)
for(i in c(14, 28, "easy", "hard")){
  load(paste("MonteCarloFrame", i,".RData", sep=" "))
  for(p in MonteCarloFrame[MonteCarloFrame$time==150,]$subject){
    temp<-data.frame(mean=sum(MonteCarloFrame[MonteCarloFrame$subject==p&MonteCarloFrame$time==150,]$currentScore)/length((MonteCarloFrame[MonteCarloFrame$subject==p&MonteCarloFrame$time==150,]$currentScore)), subject=p)
    y2<-rbind(y2,temp)
  }
}


tDataCarlo <- cleanData[cleanData$LocalTime>149,]
y14 <- data.frame(mean=0, subject=0)
for(p in tDataCarlo$SubjectNr){
  temp<-data.frame(mean=sum(tDataCarlo[tDataCarlo$SubjectNr==p&tDataCarlo$WordScore==14,]$CurrentScore)/length(tDataCarlo[tDataCarlo$SubjectNr==p&tDataCarlo$WordScore==14,]$CurrentScore), subject=p)
  y14<-rbind(y14,temp)
  }
score14test<-t.test(y14[-(1),]$mean,rep(299.5, length(y14[-(1),]$mean)),paired=TRUE) # where y1 & y2 are numeric

y28 <- data.frame(mean=0, subject=0)
for(p in tDataCarlo$SubjectNr){
  temp<-data.frame(mean=sum(tDataCarlo[tDataCarlo$SubjectNr==p&tDataCarlo$WordScore==28,]$CurrentScore)/length(tDataCarlo[tDataCarlo$SubjectNr==p&tDataCarlo$WordScore==28,]$CurrentScore), subject=p)
  y28<-rbind(y28,temp)
}
score28test<-t.test(y28[-(1),]$mean,rep(299.5, length(y28[-(1),]$mean)),paired=TRUE) # where y1 & y2 are numeric

yEasy <- data.frame(mean=0, subject=0)
for(p in tDataCarlo$SubjectNr){
  temp<-data.frame(mean=sum(tDataCarlo[tDataCarlo$SubjectNr==p&tDataCarlo$scrabbleCondition=="easy",]$CurrentScore)/length(tDataCarlo[tDataCarlo$SubjectNr==p&tDataCarlo$scrabbleCondition=="easy",]$CurrentScore), subject=p)
  yEasy<-rbind(yEasy,temp)
}
scoreEasytest<-t.test(yEasy[-(1),]$mean,rep(299.5, length(y1[-(1),]$mean)),paired=TRUE) # where y1 & y2 are numeric

yHard <- data.frame(mean=0, subject=0)
for(p in tDataCarlo$SubjectNr){
  temp<-data.frame(mean=sum(tDataCarlo[tDataCarlo$SubjectNr==p&tDataCarlo$scrabbleCondition=="hard",]$CurrentScore)/length(tDataCarlo[tDataCarlo$SubjectNr==p&tDataCarlo$scrabbleCondition=="hard",]$CurrentScore), subject=p)
  yHard<-rbind(yHard,temp)
}
scoreHardtest<-t.test(yHard[-(1),]$mean,rep(299.5, length(y1[-(1),]$mean)),paired=TRUE) # where y1 & y2 are numeric

load("MonteCarloFrame 14 .RData")
ySwitch <- data.frame(mean=0, subject=0)
for(p in MonteCarloFrame[MonteCarloFrame$time==150,]$subject){
  temp<-data.frame(mean=sum(MonteCarloFrame[MonteCarloFrame$subject==p&MonteCarloFrame$time==150,]$currentScore)/length((MonteCarloFrame[MonteCarloFrame$subject==p&MonteCarloFrame$time==150,]$currentScore)), subject=p)
  ySwitch<-rbind(ySwitch,temp)
}
switchtest<-t.test(yEasy[-(2),]$mean,rep(299.5, length(y1[-(1),]$mean)),paired=TRUE)

