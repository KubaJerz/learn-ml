studentY <- c(70, 73, 78, 79, 71, 72, 71)
yMean <- mean(studentY)
ySD <- sd(studentY)
plot(density(studentY))

studentZ <- c(98, 66, 70, 55, 52, 100, 73)
zMean <- mean(studentZ)
zSD <- sd(studentZ)
plot(density(studentZ))

#now lets normalize
yNorm <- (studentY - yMean)/ySD
zNorm <- (studentZ - zMean)/zSD

print(yNorm)
print(zNorm)
plot(density(yNorm))
plot(density(zNorm))

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



print(yNorm)
print(zNorm)
print(yBothNorm)
print(zBothnorm)

