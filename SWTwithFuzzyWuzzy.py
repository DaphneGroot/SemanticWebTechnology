import xml.etree.ElementTree as ET
import random
import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from itertools import combinations
import nltk
from nltk.translate import bleu
import numpy as np
import inflect
p = inflect.engine()

### SETTINGS
CATEGORY_SETTINGS = 'all' #run the program for all categories
PRINT_DETAILS = 'sentence' #sentence / category / overall (waterfall)
TRIPLE_AMOUNT = 1 #how many triples?

CATEGORIES = ['Airport', 'Astronaut', 'Building', 'City', 'Food', 'Monument', 'SportsTeam', 'University', 'WrittenWork']
CATEGORY = CATEGORIES[0]

if TRIPLE_AMOUNT > 1:
	TRIPLE_FILE = str(TRIPLE_AMOUNT)+'triples'
else:
	TRIPLE_FILE = str(TRIPLE_AMOUNT)+'triple_allSolutions'

#### MAIN FUNCTION
def main():
	#gather the generalized sentences per predicate of our training data
	predicateDict = train()

	overallRecall = []
	overallSentencesAmount = []
	overallSentencesCorrect = []
	overallBleuScoreList = []
	
	#loop through all categories
	if CATEGORY_SETTINGS == 'all':
		for category in CATEGORIES:
			try:
				#test the system and gather reporting values of each category
				categoryRecall, categorySentencesAmount, categorySentencesCorrect, bleuScoreList = test(category, predicateDict)
				overallRecall.append(categoryRecall)
				overallSentencesAmount.append(categorySentencesAmount)
				overallSentencesCorrect.append(categorySentencesCorrect)
				overallBleuScoreList = overallBleuScoreList + bleuScoreList
			except:
				pass
	else: 
		#test the system and gather reporting values
		categoryRecall, categorySentencesAmount, categorySentencesCorrect, bleuScoreList = test(CATEGORY, predicateDict)
		overallRecall.append(categoryRecall)
		overallSentencesAmount.append(categorySentencesAmount)
		overallSentencesCorrect.append(categorySentencesCorrect)
		overallBleuScoreList = overallBleuScoreList + bleuScoreList

	#calculate metrics
	if sum(overallRecall) > 0:
		recall = sum(overallRecall)/sum(overallSentencesAmount)
		if sum(overallSentencesCorrect) > 0:
			precision = sum(overallSentencesCorrect)/sum(overallSentencesAmount)
			fscore = 2*((precision*recall)/(precision+recall))
		else: 
			precision = 0
			fscore = 0
	else:
		precision = 0
		recall = 0
		fscore = 0

	#print metrics
	print('#'*80)
	print('#'*80)
	print('OVERALL SCORES')
	print("average Bleu score ", np.mean(overallBleuScoreList))
	print("Recall is ", sum(overallRecall), "of ", sum(overallSentencesAmount), ' (',round(recall, 3),')')
	print("sentences correct ", sum(overallSentencesCorrect), "of", sum(overallSentencesAmount), ' (',round(precision, 3),')')
	print('F-Score is ', round(fscore, 3))
	print('#'*80)
	print('#'*80)

#### TRAIN FUNCTION
def train(): 
	#read the data set files
	tree = ET.parse('WebNLG/train/'+str(TRIPLE_AMOUNT)+'triples/'+TRIPLE_FILE+'_AllExceptCC_train_challenge.xml')
	root = tree.getroot()

	predicateDict = {}
	#loop through all instances of the training set
	for i in range(len(root[0])):
		tripleSets = []
		#retrieve the tripleSets and append them to a list
		for tripleSet in root[0][i][1]:
			tripleSets.append(tripleSet.text) 

		Subjects = []
		Predicates = []
		Objects = []
		for tripleSet in tripleSets:
			Subjects.append(tripleSet.split('|')[0].strip().lower().replace("_"," "))
			Predicates.append(tripleSet.split('|')[1].strip())
			Objects.append(tripleSet.split('|')[2].strip().lower().replace("_"," ").strip("\""))
		
		#replace the Object and Subject of all possible sentences to general
		lexList = []
		for elem in root[0][i]:
			if elem.tag == 'lex':
				lexList.append(replaceToGeneral(elem.text, Subjects, Objects))

		#add the generalized sentence to the correct list in the predicateDict
		if '-'.join(Predicates) not in predicateDict:
			predicateDict['-'.join(Predicates)] = lexList
		else:
			for elem in lexList:
				if elem not in predicateDict['-'.join(Predicates)]:
					predicateDict['-'.join(Predicates)].append(elem)

		showChecks = False
		
		#only retain sentences that are adequately generalized sentences
		predicateDict = checkAndCleanCorrect(predicateDict, showChecks) #predicateDict, showCorrect Objects and Subjects = False/True

	return predicateDict

#### TEST FUNCTION
def test(category, predicateDict):
	try: 
		#read the data set files
		tree = ET.parse('WebNLG/dev/'+str(TRIPLE_AMOUNT)+'triples/'+TRIPLE_FILE+'_'+category+'_dev_challenge.xml')
		root = tree.getroot()

		bleuScoreList = []
		score = 0
		sentenceNumber = 0
		recall = 0
		#loop through all instances
		for i in range(len(root[0])):
			sentenceNumber += 1
			tripleSets = []

			#loop through all tripleSets of an instance and append these to a list
			for tripleSet in root[0][i][1]:
				tripleSets.append(tripleSet.text) 	
			Subjects = []
			Predicates = []
			Objects = []

			for tripleSet in tripleSets:
				Subjects.append(tripleSet.split('|')[0].strip().lower().replace("_"," "))
				Predicates.append(tripleSet.split('|')[1].strip())
				Objects.append(tripleSet.split('|')[2].strip().lower().replace("_"," ").strip("\""))

			# print(rootTest[0][i][1][1].text)
			n = 2
			correctSentences = []
			
			while True:
				try:
					#get all correct sentences from the test/development set, formulated by human annotators
					sentenceFromData = root[0][i][n].text
					sentenceFromData = re.sub('"', ' ', sentenceFromData)
					sentenceFromData = re.sub('\.$', '', sentenceFromData)
					sentenceFromData = re.sub('"@en', ' ', sentenceFromData)
					sentenceFromData = re.sub('\n', ' ', sentenceFromData)
					sentenceFromData = sentenceFromData.strip()
					if sentenceFromData != '':
						correctSentences.append(sentenceFromData.lower())
					n += 1
				except:
					break	
				
			#check if the predicate of the test set is in the dictionary of the training data
			if '-'.join(Predicates) in predicateDict:
				sentences = predicateDict['-'.join(Predicates)]
			else:
				sentences = [] #no predicate found...
			# print("SPO: ", Subject,Predicate, Object)
			# print("sentence: ", sentence)
			#
			ourSentences = []
			#if there are correct sentences for this predicate, formulate our generated sentence and append it to a list of ourSentences
			for sentence in sentences:
				for i in range(0, len(Subjects)):
					if i > 0:
						ourSentence = replaceToSpecific(ourSentence, Subjects[i], Objects[i], i)
					else: 
						ourSentence = replaceToSpecific(sentence, Subjects[i], Objects[i], i)

				ourSentences.append(ourSentence)

			#try to increase the recall by also inversing the predicate (if more than 1triples)
			inverseDict = {}
			for i in range(0, TRIPLE_AMOUNT):
				inverseDict[i] = TRIPLE_AMOUNT-1-i
			#Inversed check if the predicate is in the dict
			if '-'.join(Predicates[::-1]) in predicateDict:
				sentences = predicateDict['-'.join(Predicates[::-1])]
			else:
				sentences = [] #no predicate found...
			# print("SPO: ", Subject,Predicate, Object)
			# print("sentence: ", sentence)
			for sentence in sentences:
				for i in range(0, len(Subjects[::-1])):
					if i > 0:
						ourSentence = replaceToSpecific(ourSentence, Subjects[i], Objects[i], inverseDict[i])
					else: 
						ourSentence = replaceToSpecific(sentence, Subjects[i], Objects[i], inverseDict[i])

				ourSentences.append(ourSentence)

			# print("ourSentence: ", ourSentences)
			# print("correctSentences: ", correctSentences)
			ourSentences = postProcessing(ourSentences)
			
			#find the best possible sentence according to the BLEU similiraty between ourSentence and the correctSentences
			bleuScore, score, bestBleuSentence = checkCorrect(score,ourSentences,correctSentences)

			if bleuScore > 0:
				recall += 1		
				bleuScoreList.append(bleuScore)

			#report metrics per sentence
			if PRINT_DETAILS == 'sentence':
				sentence_id = str(TRIPLE_AMOUNT)+'triples_'+category+'_'+str(sentenceNumber)
				print("SENTENCE_ID \t\t\t", sentence_id)
				print("bleuScore \t\t\t", bleuScore)
				if bestBleuSentence:
					print("ourSentence \t\t\t", bestBleuSentence[1])
					print("mostSimilarCorrectSentence \t", bestBleuSentence[0])
				print('-'*80)

		if recall > 0:
			categoryRecall = recall/len(root[0])
			if score > 0:
				categoryPrecision = score/len(root[0])
				categoryFscore = 2*((categoryPrecision*categoryRecall)/(categoryPrecision+categoryRecall))
			else: 
				categoryPrecision = 0
				categoryFscore = 0
		else:
			categoryPrecision = 0
			categoryRecall = 0
			categoryFscore = 0

		#report metrics per category
		if PRINT_DETAILS == 'category' or PRINT_DETAILS == 'sentence': 
			print('~'*80)
			print('~'*80)
			print("CATEGORY: ", category)
			print("average Bleu score ", np.mean(bleuScoreList))
			print("Recall is ", recall, "of ", len(root[0]), ' (',round(categoryRecall, 3),')')
			print("sentences correct ", score, "of", len(root[0]), ' (',round(categoryPrecision, 3),')')
			print('F-Score is ', round(categoryFscore, 3))
			print('~'*80)
			print('~'*80)

		return recall, len(root[0]), score, bleuScoreList
	except:
		pass

#### CHECK BLEU SCORE
def checkCorrect(score,ourSentences,correctSentences):
	highestBleuScore = 0
	highestBleuSentence = ()
	for i in correctSentences:
		for j in ourSentences:
			if i != "" and j != "":
				correctSentence = i.split()
				ourSentence = j.split()
				#check BLEU score
				bleuScore = nltk.translate.bleu_score.sentence_bleu([correctSentence], ourSentence, weights = [1])
				#check if this sentence score is higher than the currently highest score, if so replace it
				if bleuScore > highestBleuScore:
					#print("correct with Bleu: cS", correctSentence, "with oS ", ourSentence, bleuScore)
					highestBleuScore = bleuScore
					highestBleuSentence = (i, j)

	if highestBleuScore == 1:
		score += 1
	# else:
	#  	print(ourSentences)
	#  	print(correctSentences)

	return highestBleuScore, score, highestBleuSentence

#GENERALIZE SENTENCE
def replaceToGeneral(sentence, Subjects, Objects):
	#some basic preprocessing
	sentence = sentence.lower()
	sentence = re.sub('\.$', ' ', sentence)
	sentence = re.sub('\&', 'and', sentence) 
	sentence = re.sub('"', ' ', sentence)
	sentence = re.sub('\(', 'xxx ', sentence)
	sentence = re.sub('\)', ' yyy', sentence)
	sentence = sentence.replace('u.s', 'united states')
	sentence = sentence.replace('uk', 'united kingdom')
	sentence = sentence.replace('n.y', 'new york')

	#loop through subjects and objects for substitution.
	for i in range(0, len(Subjects)):
		sentence = replaceObjectAndSubject(sentence, Subjects[i], Objects[i], i)

	return sentence
			
#GENERALIZE OBJECT AND SUBJECT
def replaceObjectAndSubject(sentence, Subject, Object, loopNumber):
	
	#some basic preprocessing
	Object = re.sub('"@en', ' ', Object)
	Object = re.sub('\"\^\^xsd\:double$', ' ', Object)
	Object = re.sub('\'\'\'\'', ' ', Object)
	Object = re.sub('\&', 'and', Object)
	Subject = re.sub('\&', 'and', Subject)
	Object = re.sub('\(', 'xxx ', Object)
	Object = re.sub('\)', ' yyy', Object)
	Subject = re.sub('\(', 'xxx ', Subject)
	Subject = re.sub('\)', ' yyy', Subject)

	###Generalize methods
	### First method: Exact replacement	
	sentence = replaceSubject(sentence, Subject, loopNumber)
	sentence = replaceObject(sentence, Object, loopNumber)

	### Second method: Replacing last comma of the subject and object
	SubjectAnd = rreplace(Subject, ',', ' and', 1)
	ObjectAnd = rreplace(Object, ',', ' and', 1)

	SubjectOr = rreplace(Subject, ',', ' or', 1)
	ObjectOr = rreplace(Object, ',', ' or', 1)

	sentence = replaceSubject(sentence, SubjectAnd, loopNumber)
	sentence = replaceObject(sentence, ObjectAnd, loopNumber)
	sentence = replaceSubject(sentence, SubjectOr, loopNumber)
	sentence = replaceObject(sentence, ObjectOr, loopNumber)

	###Third method: FC to F.C. and F.C. to FC (footballClub) or AFC and other punctuation checks

	ObjectClubThree =  re.sub(r'(\w)\.(\w)\.(\w)\.', r'\1\2\3', Object)
	ObjectClubTwo = re.sub(r'(\w)\.(\w)\.', r'\1\2', Object)
	ObjectDot = re.sub(r'(\w)\.', r'\1', Object)
	SubjectDot = re.sub(r'(\w)\.', r'\1', Subject)

	sentence = replaceObject(sentence, ObjectClubTwo, loopNumber)
	sentence = replaceObject(sentence, ObjectClubThree, loopNumber)
	sentence = replaceObject(sentence, ObjectDot, loopNumber)
	sentence = replaceSubject(sentence, SubjectDot, loopNumber)
	
	###Fourth method: Fuzzy replacement to deal with spelling errors and other differences between object/subject and the training sentences
	
	ObjectWords = Object.split()
	SubjectWords = Subject.split()
	sentenceWords = sentence.split()
	if '$subject'+str(loopNumber)+'$' not in sentence or '$object'+str(loopNumber)+'$' not in sentence:
		#print(ObjectWords)

		if '$object'+str(loopNumber)+'$' not in sentence:
			fuzzyObject = []
			#loop through all ObjectWords to match words in the sentence
			for ObjectWord in ObjectWords:
				#extract the most similar word
				word, score = process.extractOne(ObjectWord, sentenceWords, scorer=fuzz.ratio)
			
				#if the closest word has a similarity (based on levenstein) higher than 80%, this word should be replaced
				if score > 80:
					fuzzyObject.append(word)
			#check if anything in this, otherwise it would replace all whitespaces with $object$
			if len(fuzzyObject) > 0:
				sentence = replaceObject(sentence, ' '.join(fuzzyObject), loopNumber)

		if '$subject'+str(loopNumber)+'$' not in sentence:
			fuzzySubject = []
			#loop through all SubjectWords to match words in the sentence
			for SubjectWord in SubjectWords:
				#extract the most similar word
				word, score = process.extractOne(SubjectWord, sentenceWords, scorer=fuzz.ratio)			

				#if the closest word has a similarity (based on levenstein) higher than 80%, this word should be replaced
				if score > 80:
					fuzzySubject.append(word)

			#check if there is something in fuzzySubject, otherwise it would replace all whitespaces with $subject$
			if len(fuzzySubject) > 0:
				sentence = replaceSubject(sentence, ' '.join(fuzzySubject), loopNumber)
	
	# check to find all wrong sentences
	# if '$subject$' not in sentence or '$object$' not in sentence:
	# 	print(sentence + '\t '+ Object  + '\t '+  Subject)
	# check to find all correct sentences
	# if '$subject$' in sentence and '$object$'  in sentence:
	# 	print(sentence + '\t '+ Object  + '\t '+  Subject)

	return sentence

# REPLACE FUNCTIONS
def rreplace(s, old, new, occurrence):
	li = s.rsplit(old, occurrence)
	return new.join(li)

def replaceObject(sentence, Object, loopNumber):
	return re.sub(Object, '$object'+str(loopNumber)+'$', sentence)

def replaceSubject(sentence, Subject, loopNumber):
	return re.sub(Subject, '$subject'+str(loopNumber)+'$', sentence)

#CHECK GENERALIZED SENTENCES
def checkAndCleanCorrect(predicateDict, printResults = False):
#function that checks if the training sentence is adequately generalized
	i = 0
	subjects = 0
	objects = 0

	correctPredicateDict = {}

	for predicate in predicateDict:
		correctPredicateDict[predicate] = []
		for sentence in predicateDict[predicate]:
			i += 1
			if '$subject' in sentence:
				subjects += 1
			if '$object' in sentence:
				objects += 1
			if '$object' in sentence and '$subject' in sentence:
				correctPredicateDict[predicate].append(sentence)

	if printResults is True:
		print("documents: ", i)
		print("subjects: ", subjects, round(subjects*100/i, 2))
		print("objects: ", objects, round(objects*100/i, 2))

	return correctPredicateDict

#SPECIFY SENTENCE
def replaceToSpecific(sentence, Subject, Object, loopNumber):
	#some basic preprocessing
	Subject = re.sub('"@en', ' ', Subject) #Olive oil Potatoes is caused by the @en stuff...
	Object = re.sub('"@en', ' ', Object) #Olive oil Potatoes is caused by the @en stuff...
	Object = re.sub('\"\^\^xsd\:double$', ' ', Object)

	#replace the subject and object with the specific value of the test set
	sentence = sentence.replace('$subject'+str(loopNumber)+'$', Subject)
	sentence = sentence.replace('$object'+str(loopNumber)+'$', Object)

	return sentence
#post proces our formulated sentences

def postProcessing(ourSentences):
	for i, ourSentence in enumerate(ourSentences):
		ourSentence = re.sub(r'(\w*) ,', r'\1,', ourSentence) #word1 , word2 > word1, word2
		ourSentence = re.sub('xxx ', '\(', ourSentence)
		ourSentence = re.sub(' yyy', '\)', ourSentence)
		ourSentence = ourSentence.split()
		
		#indefinite clause check in the formulated sentence
		if 'a' in ourSentence or 'an' in ourSentence:
			for j, word in enumerate(ourSentence):		
				if word == 'a' or word == 'an':
					if j+1 < len(ourSentence):
							ourSentence[j] = p.a(ourSentence[j+1]).split()[0]
		ourSentences[i] = ' '.join(ourSentence)

	return ourSentences
			
if __name__ == '__main__':
	main()