import xml.etree.ElementTree as ET
import nltk
import random

def replaceToGeneral(sentence, Subject, Object):
	return sentence.lower().replace(Subject, '$subject$').replace(Object, '$object$')

def replaceToSpecific(sentence, Subject, Object):
	return sentence.replace('$subject$', Subject).replace('$object$', Object)

def testData(predicateDict):
	treeTest = ET.parse('WebNLG/dev/1triples/1triple_allSolutions_Food_dev_challenge.xml')
	rootTest = treeTest.getroot()

	for i in range(3):#range(len(root[0])):
		tripleSet = rootTest[0][i][1][0].text
		Subject   = tripleSet.split('|')[0].strip().lower().replace("_"," ")
		Predicate = tripleSet.split('|')[1].strip()
		Object    = tripleSet.split('|')[2].strip().lower().replace("_"," ").strip("_\"")

		sentence = random.choice(predicateDict[Predicate])
		# sentence = predicateDict[Predicate][1]
		print(replaceToSpecific(sentence,Subject,Object))

def main():
	tree = ET.parse('WebNLG/train/1triples/1triple_allSolutions_Food_train_challenge.xml')
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

	#print(predicateDict)

	testData(predicateDict)


if __name__ == '__main__':
	main()