'''
NORMALIZATION ABYSS

So whats all this for? Well we are in R because you need to keep yourseld on your toes and try new things!
But now for the normalization abyss:

Our very first lesson is that you need to normalize the inputs to your NN  but I want to go a bit deeper here

So normalization of inputs to a NN is long been know to speed up training (LeCun et al. 1998)
So then Batch Norm comes around in 2015 (Ioffe Szegedy 2015) and they show "empiricly" 
that normalizing in the hidden layers is benifical. (pre or post activation is another disscion)

I will not go into the "reasoning" or thoughts behind this but I have a very simple example 
with the idea to warn you! Normalization is a great tool but needs to be understud because 
if done wrong it can royaly mess things up.

'''

#Studet Y is not the best student they just dont care in general and don't try in school
studentY <- c(70, 73, 78, 79, 71, 72, 71)
yMean <- mean(studentY)
ySD <- sd(studentY)
plot(density(studentY))

#Now we have student Z by looking at the grades we can tell that they struggle to keep up 
#in some subjects but there is some subject say art that they are amazing in 
studentZ <- c(98, 66, 70, 55, 52, 100, 73)
zMean <- mean(studentZ)
zSD <- sd(studentZ)
plot(density(studentZ))

#This is a bit of a consturde example but now lets normalize
yNorm <- (studentY - yMean)/ySD
zNorm <- (studentZ - zMean)/zSD

print(yNorm)
print(zNorm)
plot(density(yNorm))
plot(density(zNorm))

#the way the grades look in relationship to each other is now totally differently
#So you say well Kuba this is obvious if they have some relationship you would not normalize them separately
#Thats correct but while its obvous here it becomes less intuative when you are in the middel of a NN
#All i say is Normalize with caution

#new mean and sd
yNormMean <- mean(yNorm)
yNormSD <- sd(yNorm)

zNormMean <- mean(zNorm)
zNormSD <- sd(zNorm)

bothMean <- mean(c(studentY, studentZ))
bothSD <- sd(c(studentY, studentZ))

yBothNorm <- (studentY - bothMean)/bothSD
zBothnorm <- (studentZ - bothMean)/bothSD
  
plot(density(yBothNorm))
plot(density(zBothnorm))
#So now we normalized with their combnined Mean and sd


print(yNorm)
print(zNorm)
print(yBothNorm)
print(zBothnorm)

