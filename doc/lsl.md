# LSL commands

Commands should be send over LSL using a single channel holding a single string of commands. There are currently four commands, see below, each followed by parameters.

``` text
create;<name>;<type>;<time>;<filter>;<method>
collect;<name>;<class>
train;<name>
predict;<name>
```

examples:

``` text
create;workload;LDA;[-400,600];[1,30];windowed-average-EEG
collect;workload;0
collect;workload;1
train;workload
predict;workload
```

``` text
create;relaxation;SVM;[-400,600];[8,30];windowed-average-EEG-plus-headmovement
collect;relaxation;0
collect;relaxation;1
train;relaxation
predict;relaxation
```

``` text
create;relaxation;LDA;[-400,600];[1,30];filter-bank-alpha-theta-plus-headmovement
collect;relaxation;0
collect;relaxation;1
train;relaxation
predict;relaxation
```
