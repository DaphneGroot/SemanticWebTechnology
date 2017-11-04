import xml.etree.ElementTree as ET
import nltk
import random
import re
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from itertools import combinations
from nltk.corpus import wordnet as wn
from nltk.translate import bleu
import numpy as np

def checkCorrect(score,ourSentences,correctSentences):
	highestBleuScore = 0
	for i in correctSentences:
		for j in ourSentences:
			if i != "" and j != "":
				correctSentence = i.split()
				ourSentence = j.split()
				bleuScore = nltk.translate.bleu_score.sentence_bleu([correctSentence], ourSentence, weights = [1])

				if bleuScore > highestBleuScore:
					print("correct with Bleu: cS", correctSentence, "with oS ", ourSentence, bleuScore)
					highestBleuScore = bleuScore


	if highestBleuScore == 1:
		score += 1



	return highestBleuScore, score



def checkAndCleanCorrect(predicateDict, printResults = False):
	i = 0
	subjects = 0
	objects = 0

	correctPredicateDict = {}

	for predicate in predicateDict:
		correctPredicateDict[predicate] = []
		for sentence in predicateDict[predicate]:
			i += 1
			if '$subject$' in sentence:
				subjects += 1
			if '$object$' in sentence:
				objects += 1
			if '$object$' in sentence and '$subject$' in sentence:
				correctPredicateDict[predicate].append(sentence)

	if printResults is True:
		print("documents: ", i)
		print("subjects: ", subjects, round(subjects*100/i, 2))
		print("objects: ", objects, round(objects*100/i, 2))

	return correctPredicateDict

def getDerivatives(word): #not used now
	allForms = []
	synsets = wn.synsets(word)
	# Word not found
	if not synsets:
		return []

    # Get all  lemmas of the word
	lemmas = [l for s in synsets for l in s.lemmas()]

	# Get related forms
	derivationally_related_forms = [l.derivationally_related_forms() for l in lemmas]

	for form in derivationally_related_forms:
		for elem in form:
			if elem.name() not in allForms:
				allForms.append(elem.name())

	return allForms

def rreplace(s, old, new, occurrence):
	li = s.rsplit(old, occurrence)
	return new.join(li)

def replaceObject(Object, sentence):
	return re.sub(Object + '(\s)', '$object$ ', sentence)

def replaceSubject(Subject, sentence):
	return re.sub(Subject + '(\s)', '$subject$ ', sentence)
		
def replaceToGeneral(sentence, Subject, Object):
	sentence = sentence.lower()
	sentence = re.sub('\.$', ' ', sentence)

	Object = re.sub('"@en', ' ', Object) #Olive oil Potatoes is caused by the @en stuff...

	ObjectWords = Object.split()
	SubjectWords = Subject.split()
	sentenceWords = sentence.split()

	sentence = sentence.replace('u.s', 'united states')
	sentence = sentence.replace('uk', 'united kingdom')

	### FIRST METHOD: Exact replacement	
	sentence = replaceSubject(Subject, sentence)
	sentence = replaceObject(Object, sentence)

	### SECOND METHOD: REPLACING LAST COMMA WITH SOMETHING ELSE
	SubjectAnd = rreplace(Subject, ',', ' and', 1)
	ObjectAnd = rreplace(Object, ',', ' and', 1)

	SubjectOr = rreplace(Subject, ',', ' or', 1)
	ObjectOr = rreplace(Object, ',', ' or', 1)

	sentence = replaceSubject(SubjectAnd, sentence)
	sentence = replaceObject(ObjectAnd, sentence)
	sentence = replaceSubject(SubjectOr, sentence)
	sentence = replaceObject(ObjectOr, sentence)

	### THIRD METHOD: Fuzzy replacement
	if '$subject$' not in sentence or '$object$' not in sentence:
		#print(ObjectWords)
		if '$object$' not in sentence:
			fuzzyObject = []
			#loop through all ObjectWords to match words in the sentence
			for ObjectWord in ObjectWords:
				word, score = process.extractOne(ObjectWord, sentenceWords, scorer=fuzz.ratio)
				if score > 80:
					fuzzyObject.append(word)
			#check if anything in this, otherwise it would replace all whitespaces with $object$
			if len(fuzzyObject) > 0:
				sentence = replaceObject(' '.join(fuzzyObject), sentence)

		if '$subject$' not in sentence:
			fuzzySubject = []
			#loop through all SubjectWords to match words in the sentence
			for SubjectWord in SubjectWords:
				word, score = process.extractOne(SubjectWord, sentenceWords, scorer=fuzz.ratio)
				if score > 80:
					fuzzySubject.append(word)

			#check if anything in this, otherwise it would replace all whitespaces with $subject$
			if len(fuzzySubject) > 0:
				sentence = replaceSubject(' '.join(fuzzySubject), sentence)

	#check to find all wrong sentences
	#if '$subject$' not in sentence or '$object$' not in sentence:
		#print(sentence + '\t '+ Object  + '\t '+  Subject)
	#check to find all correct sentences
	#if '$subject$' in sentence and '$object$'  in sentence:
	#	print(sentence + '\t '+ Object  + '\t '+  Subject)
	return sentence
	

def showPredicateRecall(predicateDict, minimumAmount = 1):
	i = 0
	for predicate in predicateDict:
		if len(predicateDict[predicate]) >= 1:
			i += 1
	print("Out of the {} predicates, {} have at least {} syntactical sentence(s) ".format(len(predicateDict), i, minimumAmount))

def replaceToSpecific(sentence, Subject, Object):
	return sentence.replace('$subject$', Subject).replace('$object$', Object)

def testData(predicateDict):
	bleuScoreList = []

	score = 0
	treeTest = ET.parse('WebNLG/dev/1triples/1triple_allSolutions_Building_dev_challenge.xml')
	rootTest = treeTest.getroot()
	for i in range(len(rootTest[0])):
		tripleSet = rootTest[0][i][1][0].text
		Subject   = tripleSet.split('|')[0].strip().lower().replace("_"," ")
		Predicate = tripleSet.split('|')[1].strip()
		Object    = tripleSet.split('|')[2].strip().lower().replace("_"," ").strip("_\"")

		#to obtain the sentences from the XML files
		n = 2
		correctSentences = []
		while True:
			try:
				sentenceFromData = rootTest[0][i][n].text
				sentenceFromData = re.sub('"', ' ', sentenceFromData)
				sentenceFromData = re.sub('\.$', '', sentenceFromData)
				sentenceFromData = re.sub('"@en', ' ', sentenceFromData)
				sentenceFromData = re.sub('\n', ' ', sentenceFromData)
				sentenceFromData = sentenceFromData.strip()
				correctSentences.append(sentenceFromData.lower())
				n += 1
			except:
				break

		
		#check if the predicate is in the dict
		if Predicate in predicateDict:
			sentence = predicateDict[Predicate]
		else:
			sentence = "" #just an ugly message

		
		# print("SPO: ", Subject,Predicate, Object)
		# print("sentence: ", sentence)
		ourSentences = []
		for s in sentence:
			ourSentence = replaceToSpecific(s,Subject,Object)
			ourSentences.append(ourSentence)

		# print("ourSentence: ", ourSentences)
		# print("correctSentences: ", correctSentences)
		
		bleuScore, score = checkCorrect(score,ourSentences,correctSentences)
		
		print('-'*50)

		bleuScoreList.append(bleuScore)

	print("average Bleu score ", np.mean(bleuScoreList))
	print("sentences correct ", score, "of", len(rootTest[0]))


def main():
	tree = ET.parse('WebNLG/train/1triples/1triple_allSolutions_Building_train_challenge.xml')
	root = tree.getroot()
	predicateDict = {}

	for i in range(len(root[0])):
		
		tripleSet = root[0][i][1][0].text
		Subject   = tripleSet.split('|')[0].strip().lower().replace("_"," ")
		Predicate = tripleSet.split('|')[1].strip()
		Object    = tripleSet.split('|')[2].strip().lower().replace("_"," ").strip("\"")


		lexList = []
		for elem in root[0][i]:
			if elem.tag == 'lex':
				lexList.append(replaceToGeneral(elem.text, Subject, Object))

		if Predicate not in predicateDict:
			predicateDict[Predicate] = lexList
		else:
			for elem in lexList:
				if elem not in predicateDict[Predicate]:
					predicateDict[Predicate].append(elem)
	showChecks = True
	predicateDict = checkAndCleanCorrect(predicateDict, showChecks) #predicateDict, showCorrect Objects and Subjects = False/True
	if showChecks is True:
		showPredicateRecall(predicateDict, 1)
	
	testData(predicateDict)

if __name__ == '__main__':
	main()