# Dependencies

```
g++ (C++23)
Make
Apache Arrow/Parquet - https://arrow.apache.org/install/
```

## Build instructions

```
mkdir bin
make
```

## Example usage

```
$ bin/llm
crunching numbers...
unique n grams: 114885
loaded another chunk (1/1191582)
...
loaded another chunk (1191581/1191582)
finished collecting soda utterances 1191582
finished collecting dialog csv utterances 3725
total utterances count 1195307
unique n grams: 3050343
training time: 363.396s
Prompt: How are you ?
history you|?|<stop>
history ?|<stop>|Not
history <stop>|Not|really
history Not|really|.
history really|.|I'm
history .|I'm|just
history I'm|just|really
history just|really|frustrated
history really|frustrated|with
history frustrated|with|my
history with|my|job
history my|job|.
Prediction: Not really . I'm just really frustrated with my job . <stop> 
Prompt: I'm sorry to hear that .
history that|.|<stop>
history .|<stop>|Oh,
history <stop>|Oh,|trust
history Oh,|trust|me,
history trust|me,|there
history me,|there|is
history there|is|always
history is|always|a
history always|a|good
history a|good|feeling
history good|feeling|about
history feeling|about|it
history about|it|.
history it|.|It's
history .|It's|been
history It's|been|so
history been|so|nice
history so|nice|out
history nice|out|.
Prediction: Oh, trust me, there is always a good feeling about it . It's been so nice out . <stop> 
Prompt: Do you understand what you're saying ?
history saying|?|<stop>
history ?|<stop>|That
history <stop>|That|sounds
history That|sounds|like
history sounds|like|it's
history like|it's|going
history it's|going|to
history going|to|be
history to|be|a
history be|a|great
history a|great|way
history great|way|to
history way|to|capture
history to|capture|moments
history capture|moments|and
history moments|and|create
vocab miss
history and|create|be
vocab miss
history create|be|did
vocab miss
history be|did|need
vocab miss
history did|need|to
vocab miss
history need|to|seriously
history to|seriously|consider
vocab miss
history seriously|consider|saw
vocab miss
history consider|saw|isn't
vocab miss
history saw|isn't|really
vocab miss
history isn't|really|and
vocab miss
history really|and|if
vocab miss
history and|if|they're
history if|they're|not
history they're|not|what
history not|what|we
history what|we|need
history we|need|to
history need|to|be
history to|be|more
history be|more|intimate
history more|intimate|and
history intimate|and|special
history and|special|.
Prediction: That sounds like it's going to be a great way to capture moments and create be did need to seriously consider saw isn't really and if they're not what we need to be more intimate and special . <stop>
```