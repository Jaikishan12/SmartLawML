def getInstancesAndTokesForArgumentBy():
    from smartlawdata import getArgByDataSet
    from nltk.tokenize import word_tokenize
    from nltk.probability import FreqDist
    arg_by = getArgByDataSet()
    arg_by_applicant = arg_by[arg_by['argumentBy']=='APPLICANT']
    arg_by_applicant = arg_by_applicant.apply(' '.join, axis=0)
    arg_by_applicant = arg_by_applicant.apply(word_tokenize)
    fdist = FreqDist(arg_by_applicant[0])    
    vect=fdist.values()    
    out = [{'elementType':'Argument By','label':'APPLICANT','tokens':len(fdist),'instances':sum(vect)}]
    
    arg_by_respondent = arg_by[arg_by['argumentBy']=='RESPONDENT']
    arg_by_respondent = arg_by_respondent.apply(' '.join, axis=0)
    arg_by_respondent = arg_by_respondent.apply(word_tokenize)
    fdist = FreqDist(arg_by_respondent[0])    
    vect=fdist.values()    
    out += [{'elementType':'Argument By','label':'RESPONDENT','tokens':len(fdist),'instances':sum(vect)}]
    
    arg_by_judge = arg_by[arg_by['argumentBy']=='JUDGE']
    arg_by_judge = arg_by_judge.apply(' '.join, axis=0)
    arg_by_judge = arg_by_judge.apply(word_tokenize)
    fdist = FreqDist(arg_by_judge[0])    
    vect=fdist.values()    
    out += [{'elementType':'Argument By','label':'JUDGE','tokens':len(fdist),'instances':sum(vect)}]
    
    return out

def getInstancesAndTokesForSentType():
    from smartlawdata import getSentenceTypeDataSet
    from nltk.tokenize import word_tokenize
    from nltk.probability import FreqDist
    sent_type = getSentenceTypeDataSet()
    sent_type_premise = sent_type[sent_type['argumentSentenceType']=='PREMISE']
    sent_type_premise = sent_type_premise.apply(' '.join, axis=0)
    sent_type_premise = sent_type_premise.apply(word_tokenize)
    fdist = FreqDist(sent_type_premise[0])    
    vect=fdist.values()    
    out = [{'elementType':'Sentence Type','label':'PREMISE','tokens':len(fdist),'instances':sum(vect)}]
    
    sent_type_conclusion = sent_type[sent_type['argumentSentenceType']=='CONCLUSION']
    sent_type_conclusion = sent_type_conclusion.apply(' '.join, axis=0)
    sent_type_conclusion = sent_type_conclusion.apply(word_tokenize)
    fdist = FreqDist(sent_type_conclusion[0])    
    vect=fdist.values()    
    out += [{'elementType':'Sentence Type','label':'CONCLUSION','tokens':len(fdist),'instances':sum(vect)}]
    
    sent_type_na = sent_type[sent_type['argumentSentenceType']=='NA']
    sent_type_na = sent_type_na.apply(' '.join, axis=0)
    sent_type_na = sent_type_na.apply(word_tokenize)
    fdist = FreqDist(sent_type_na[0])    
    vect=fdist.values()    
    out += [{'elementType':'Sentence Type','label':'NA','tokens':len(fdist),'instances':sum(vect)}]
    
    return out

def getInstancesAndTokesForOrderType():
    from smartlawdata import getOrderTypeDataSet
    from nltk.tokenize import word_tokenize
    from nltk.probability import FreqDist
    order_type = getOrderTypeDataSet()
    order_type_accepted = order_type[order_type['orderType']=='ACCEPTED']
    order_type_accepted = order_type_accepted.apply(' '.join, axis=0)
    order_type_accepted = order_type_accepted.apply(word_tokenize)
    fdist = FreqDist(order_type_accepted[0])    
    vect=fdist.values()    
    out = [{'elementType':'Order Type','label':'ACCEPTED','tokens':len(fdist),'instances':sum(vect)}]
    
    order_type_rejected = order_type[order_type['orderType']=='REJECTED']
    order_type_rejected = order_type_rejected.apply(' '.join, axis=0)
    order_type_rejected = order_type_rejected.apply(word_tokenize)
    fdist = FreqDist(order_type_rejected[0])    
    vect=fdist.values()    
    out += [{'elementType':'Order Type','label':'REJECTED','tokens':len(fdist),'instances':sum(vect)}]
    
    return out

def getDatasetStatistics():
    out = getInstancesAndTokesForArgumentBy()
    out += getInstancesAndTokesForSentType()
    out += getInstancesAndTokesForOrderType()
    return out