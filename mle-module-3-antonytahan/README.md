# MiniTorch Module 3

<img src="https://minitorch.github.io/minitorch.svg" width="50%">

* Docs: https://minitorch.github.io/

* Overview: https://minitorch.github.io/module3.html


You will need to modify `tensor_functions.py` slightly in this assignment.

* Tests:

```
python run_tests.py
```


# CPU Simple Dataset:
```
Epoch  0  time  15.673148155212402
Epoch  0  loss  6.559663740895859 correct 37
Average time per epoch 0.03134629631042481 (for 500 epochs)
Epoch  1  time  0.14148712158203125
Average time per epoch 0.03162927055358887 (for 500 epochs)
Epoch  2  time  0.14731478691101074
Average time per epoch 0.031923900127410886 (for 500 epochs)
Epoch  3  time  0.21529078483581543
Average time per epoch 0.03235448169708252 (for 500 epochs)
Epoch  4  time  0.14655590057373047
Average time per epoch 0.03264759349822998 (for 500 epochs)
Epoch  5  time  0.14657092094421387
Average time per epoch 0.03294073534011841 (for 500 epochs)
Epoch  6  time  0.1943659782409668
Average time per epoch 0.03332946729660034 (for 500 epochs)
Epoch  7  time  0.14489030838012695
Average time per epoch 0.03361924791336059 (for 500 epochs)
Epoch  8  time  0.13997888565063477
Average time per epoch 0.033899205684661864 (for 500 epochs)
Epoch  9  time  0.1443617343902588
Average time per epoch 0.03418792915344238 (for 500 epochs)
Epoch  10  time  0.14350175857543945
Epoch  10  loss  3.681482561227548 correct 46
Average time per epoch 0.03447493267059326 (for 500 epochs)
Epoch  11  time  0.15019989013671875
Average time per epoch 0.0347753324508667 (for 500 epochs)
Epoch  12  time  0.14380502700805664
Average time per epoch 0.035062942504882816 (for 500 epochs)
Epoch  13  time  0.1625072956085205
Average time per epoch 0.035387957096099855 (for 500 epochs)
Epoch  14  time  0.14612078666687012
Average time per epoch 0.035680198669433595 (for 500 epochs)
Epoch  15  time  0.150071382522583
Average time per epoch 0.03598034143447876 (for 500 epochs)
Epoch  16  time  0.1443638801574707
Average time per epoch 0.0362690691947937 (for 500 epochs)
Epoch  17  time  0.149505615234375
Average time per epoch 0.03656808042526245 (for 500 epochs)
Epoch  18  time  0.14438867568969727
Average time per epoch 0.03685685777664185 (for 500 epochs)
Epoch  19  time  0.15319013595581055
Average time per epoch 0.03716323804855347 (for 500 epochs)
Epoch  20  time  0.15611791610717773
Epoch  20  loss  0.8880816748458324 correct 46
Average time per epoch 0.03747547388076782 (for 500 epochs)
Epoch  21  time  0.14636898040771484
Average time per epoch 0.03776821184158325 (for 500 epochs)
Epoch  22  time  0.14259696006774902
Average time per epoch 0.03805340576171875 (for 500 epochs)
Epoch  23  time  0.14499998092651367
Average time per epoch 0.038343405723571776 (for 500 epochs)
Epoch  24  time  0.14327335357666016
Average time per epoch 0.0386299524307251 (for 500 epochs)
Epoch  25  time  0.14966750144958496
Average time per epoch 0.03892928743362427 (for 500 epochs)
Epoch  26  time  0.1451282501220703
Average time per epoch 0.03921954393386841 (for 500 epochs)
Epoch  27  time  0.16534686088562012
Average time per epoch 0.03955023765563965 (for 500 epochs)
Epoch  28  time  0.139754056930542
Average time per epoch 0.039829745769500734 (for 500 epochs)
Epoch  29  time  0.15035057067871094
Average time per epoch 0.04013044691085815 (for 500 epochs)
Epoch  30  time  0.1417527198791504
Epoch  30  loss  1.6089890923063463 correct 47
Average time per epoch 0.040413952350616456 (for 500 epochs)
Epoch  31  time  0.1497058868408203
Average time per epoch 0.040713364124298095 (for 500 epochs)
Epoch  32  time  0.14221954345703125
Average time per epoch 0.04099780321121216 (for 500 epochs)
Epoch  33  time  0.14178061485290527
Average time per epoch 0.04128136444091797 (for 500 epochs)
Epoch  34  time  0.1655135154724121
Average time per epoch 0.04161239147186279 (for 500 epochs)
Epoch  35  time  0.13913774490356445
Average time per epoch 0.041890666961669924 (for 500 epochs)
Epoch  36  time  0.15266776084899902
Average time per epoch 0.04219600248336792 (for 500 epochs)
Epoch  37  time  0.14328932762145996
Average time per epoch 0.042482581138610837 (for 500 epochs)
Epoch  38  time  0.14394903182983398
Average time per epoch 0.042770479202270506 (for 500 epochs)
Epoch  39  time  0.15531206130981445
Average time per epoch 0.04308110332489014 (for 500 epochs)
Epoch  40  time  0.16278433799743652
Epoch  40  loss  1.0113719941294075 correct 49
Average time per epoch 0.04340667200088501 (for 500 epochs)
Epoch  41  time  0.14279842376708984
Average time per epoch 0.04369226884841919 (for 500 epochs)
Epoch  42  time  0.14833736419677734
Average time per epoch 0.043988943576812746 (for 500 epochs)
Epoch  43  time  0.14495134353637695
Average time per epoch 0.0442788462638855 (for 500 epochs)
Epoch  44  time  0.15935397148132324
Average time per epoch 0.04459755420684815 (for 500 epochs)
Epoch  45  time  0.14693808555603027
Average time per epoch 0.04489143037796021 (for 500 epochs)
Epoch  46  time  0.1500842571258545
Average time per epoch 0.045191598892211915 (for 500 epochs)
Epoch  47  time  0.17005324363708496
Average time per epoch 0.04553170537948609 (for 500 epochs)
Epoch  48  time  0.14841556549072266
Average time per epoch 0.04582853651046753 (for 500 epochs)
Epoch  49  time  0.15018486976623535
Average time per epoch 0.04612890625 (for 500 epochs)
Epoch  50  time  0.1641407012939453
Epoch  50  loss  1.4302956480960562 correct 48
Average time per epoch 0.04645718765258789 (for 500 epochs)
Epoch  51  time  0.16413283348083496
Average time per epoch 0.04678545331954956 (for 500 epochs)
Epoch  52  time  0.15970945358276367
Average time per epoch 0.04710487222671509 (for 500 epochs)
Epoch  53  time  0.17046594619750977
Average time per epoch 0.047445804119110105 (for 500 epochs)
Epoch  54  time  0.15645456314086914
Average time per epoch 0.04775871324539185 (for 500 epochs)
Epoch  55  time  0.14516234397888184
Average time per epoch 0.04804903793334961 (for 500 epochs)
Epoch  56  time  0.14958977699279785
Average time per epoch 0.048348217487335206 (for 500 epochs)
Epoch  57  time  0.14940762519836426
Average time per epoch 0.04864703273773193 (for 500 epochs)
Epoch  58  time  0.1598057746887207
Average time per epoch 0.048966644287109375 (for 500 epochs)
Epoch  59  time  0.14625334739685059
Average time per epoch 0.049259150981903074 (for 500 epochs)
Epoch  60  time  0.15928101539611816
Epoch  60  loss  1.1800221353877334 correct 48
Average time per epoch 0.04957771301269531 (for 500 epochs)
Epoch  61  time  0.14169573783874512
Average time per epoch 0.049861104488372804 (for 500 epochs)
Epoch  62  time  0.14495348930358887
Average time per epoch 0.05015101146697998 (for 500 epochs)
Epoch  63  time  0.14778685569763184
Average time per epoch 0.05044658517837525 (for 500 epochs)
Epoch  64  time  0.1464242935180664
Average time per epoch 0.050739433765411374 (for 500 epochs)
Epoch  65  time  0.14443039894104004
Average time per epoch 0.051028294563293455 (for 500 epochs)
Epoch  66  time  0.1498558521270752
Average time per epoch 0.051328006267547605 (for 500 epochs)
Epoch  67  time  0.15331053733825684
Average time per epoch 0.05163462734222412 (for 500 epochs)
Epoch  68  time  0.15075325965881348
Average time per epoch 0.05193613386154175 (for 500 epochs)
Epoch  69  time  0.14110016822814941
Average time per epoch 0.05221833419799805 (for 500 epochs)
Epoch  70  time  0.15706300735473633
Epoch  70  loss  0.8821541940351437 correct 49
Average time per epoch 0.05253246021270752 (for 500 epochs)
Epoch  71  time  0.1446666717529297
Average time per epoch 0.05282179355621338 (for 500 epochs)
Epoch  72  time  0.14730429649353027
Average time per epoch 0.05311640214920044 (for 500 epochs)
Epoch  73  time  0.14355683326721191
Average time per epoch 0.05340351581573486 (for 500 epochs)
Epoch  74  time  0.18924260139465332
Average time per epoch 0.05378200101852417 (for 500 epochs)
Epoch  75  time  0.14229202270507812
Average time per epoch 0.05406658506393432 (for 500 epochs)
Epoch  76  time  0.15447473526000977
Average time per epoch 0.054375534534454345 (for 500 epochs)
Epoch  77  time  0.13941049575805664
Average time per epoch 0.054654355525970456 (for 500 epochs)
Epoch  78  time  0.15055012702941895
Average time per epoch 0.0549554557800293 (for 500 epochs)
Epoch  79  time  0.14261341094970703
Average time per epoch 0.05524068260192871 (for 500 epochs)
Epoch  80  time  0.15541410446166992
Epoch  80  loss  1.1082150213776902 correct 49
Average time per epoch 0.05555151081085205 (for 500 epochs)
Epoch  81  time  0.14141631126403809
Average time per epoch 0.055834343433380125 (for 500 epochs)
Epoch  82  time  0.1472492218017578
Average time per epoch 0.056128841876983644 (for 500 epochs)
Epoch  83  time  0.1429462432861328
Average time per epoch 0.05641473436355591 (for 500 epochs)
Epoch  84  time  0.15028715133666992
Average time per epoch 0.05671530866622925 (for 500 epochs)
Epoch  85  time  0.1478416919708252
Average time per epoch 0.0570109920501709 (for 500 epochs)
Epoch  86  time  0.14669179916381836
Average time per epoch 0.05730437564849854 (for 500 epochs)
Epoch  87  time  0.15727543830871582
Average time per epoch 0.057618926525115965 (for 500 epochs)
Epoch  88  time  0.1454484462738037
Average time per epoch 0.057909823417663576 (for 500 epochs)
Epoch  89  time  0.1513674259185791
Average time per epoch 0.05821255826950073 (for 500 epochs)
Epoch  90  time  0.15427875518798828
Epoch  90  loss  0.4852628307685724 correct 48
Average time per epoch 0.05852111577987671 (for 500 epochs)
Epoch  91  time  0.14362382888793945
Average time per epoch 0.05880836343765259 (for 500 epochs)
Epoch  92  time  0.1530921459197998
Average time per epoch 0.059114547729492185 (for 500 epochs)
Epoch  93  time  0.14240384101867676
Average time per epoch 0.05939935541152954 (for 500 epochs)
Epoch  94  time  0.17026066780090332
Average time per epoch 0.05973987674713135 (for 500 epochs)
Epoch  95  time  0.14648818969726562
Average time per epoch 0.06003285312652588 (for 500 epochs)
Epoch  96  time  0.14811420440673828
Average time per epoch 0.060329081535339356 (for 500 epochs)
Epoch  97  time  0.14429259300231934
Average time per epoch 0.06061766672134399 (for 500 epochs)
Epoch  98  time  0.14678716659545898
Average time per epoch 0.06091124105453491 (for 500 epochs)
Epoch  99  time  0.14105749130249023
Average time per epoch 0.06119335603713989 (for 500 epochs)
Epoch  100  time  0.14519977569580078
Epoch  100  loss  0.7155429412816158 correct 49
Average time per epoch 0.061483755588531495 (for 500 epochs)
Epoch  101  time  0.1569366455078125
Average time per epoch 0.061797628879547116 (for 500 epochs)
Epoch  102  time  0.15005993843078613
Average time per epoch 0.062097748756408694 (for 500 epochs)
Epoch  103  time  0.1461777687072754
Average time per epoch 0.062390104293823245 (for 500 epochs)
Epoch  104  time  0.15337467193603516
Average time per epoch 0.06269685363769531 (for 500 epochs)
Epoch  105  time  0.14257574081420898
Average time per epoch 0.06298200511932373 (for 500 epochs)
Epoch  106  time  0.14722919464111328
Average time per epoch 0.06327646350860595 (for 500 epochs)
Epoch  107  time  0.1451416015625
Average time per epoch 0.06356674671173096 (for 500 epochs)
Epoch  108  time  0.15775179862976074
Average time per epoch 0.06388225030899047 (for 500 epochs)
Epoch  109  time  0.14986276626586914
Average time per epoch 0.06418197584152222 (for 500 epochs)
Epoch  110  time  0.14716434478759766
Epoch  110  loss  0.610253532890327 correct 49
Average time per epoch 0.06447630453109741 (for 500 epochs)
Epoch  111  time  0.1441183090209961
Average time per epoch 0.0647645411491394 (for 500 epochs)
Epoch  112  time  0.14679837226867676
Average time per epoch 0.06505813789367676 (for 500 epochs)
Epoch  113  time  0.1409618854522705
Average time per epoch 0.0653400616645813 (for 500 epochs)
Epoch  114  time  0.149810791015625
Average time per epoch 0.06563968324661255 (for 500 epochs)
Epoch  115  time  0.1587378978729248
Average time per epoch 0.0659571590423584 (for 500 epochs)
Epoch  116  time  0.15446019172668457
Average time per epoch 0.06626607942581177 (for 500 epochs)
Epoch  117  time  0.14127516746520996
Average time per epoch 0.06654862976074219 (for 500 epochs)
Epoch  118  time  0.14842438697814941
Average time per epoch 0.06684547853469848 (for 500 epochs)
Epoch  119  time  0.14365744590759277
Average time per epoch 0.06713279342651367 (for 500 epochs)
Epoch  120  time  0.1535942554473877
Epoch  120  loss  0.5803415886545561 correct 48
Average time per epoch 0.06743998193740845 (for 500 epochs)
Epoch  121  time  0.1599287986755371
Average time per epoch 0.06775983953475952 (for 500 epochs)
Epoch  122  time  0.16324663162231445
Average time per epoch 0.06808633279800415 (for 500 epochs)
Epoch  123  time  0.163010835647583
Average time per epoch 0.06841235446929932 (for 500 epochs)
Epoch  124  time  0.1614823341369629
Average time per epoch 0.06873531913757325 (for 500 epochs)
Epoch  125  time  0.15261578559875488
Average time per epoch 0.06904055070877076 (for 500 epochs)
Epoch  126  time  0.14231467247009277
Average time per epoch 0.06932518005371094 (for 500 epochs)
Epoch  127  time  0.14743423461914062
Average time per epoch 0.06962004852294922 (for 500 epochs)
Epoch  128  time  0.15657877922058105
Average time per epoch 0.06993320608139038 (for 500 epochs)
Epoch  129  time  0.1621248722076416
Average time per epoch 0.07025745582580567 (for 500 epochs)
Epoch  130  time  0.14988398551940918
Epoch  130  loss  2.7789807665709083 correct 47
Average time per epoch 0.07055722379684448 (for 500 epochs)
Epoch  131  time  0.15204954147338867
Average time per epoch 0.07086132287979126 (for 500 epochs)
Epoch  132  time  0.1484394073486328
Average time per epoch 0.07115820169448853 (for 500 epochs)
Epoch  133  time  0.15099000930786133
Average time per epoch 0.07146018171310425 (for 500 epochs)
Epoch  134  time  0.15239357948303223
Average time per epoch 0.07176496887207032 (for 500 epochs)
Epoch  135  time  0.16556763648986816
Average time per epoch 0.07209610414505005 (for 500 epochs)
Epoch  136  time  0.14978504180908203
Average time per epoch 0.07239567422866822 (for 500 epochs)
Epoch  137  time  0.14272260665893555
Average time per epoch 0.07268111944198609 (for 500 epochs)
Epoch  138  time  0.14794540405273438
Average time per epoch 0.07297701025009155 (for 500 epochs)
Epoch  139  time  0.13805747032165527
Average time per epoch 0.07325312519073486 (for 500 epochs)
Epoch  140  time  0.15337371826171875
Epoch  140  loss  2.0100597937154245 correct 49
Average time per epoch 0.0735598726272583 (for 500 epochs)
Epoch  141  time  0.15839695930480957
Average time per epoch 0.07387666654586791 (for 500 epochs)
Epoch  142  time  0.14878630638122559
Average time per epoch 0.07417423915863038 (for 500 epochs)
Epoch  143  time  0.16183972358703613
Average time per epoch 0.07449791860580444 (for 500 epochs)
Epoch  144  time  0.151444673538208
Average time per epoch 0.07480080795288085 (for 500 epochs)
Epoch  145  time  0.14107561111450195
Average time per epoch 0.07508295917510986 (for 500 epochs)
Epoch  146  time  0.14822816848754883
Average time per epoch 0.07537941551208496 (for 500 epochs)
Epoch  147  time  0.1402294635772705
Average time per epoch 0.07565987443923951 (for 500 epochs)
Epoch  148  time  0.16629767417907715
Average time per epoch 0.07599246978759766 (for 500 epochs)
Epoch  149  time  0.1532299518585205
Average time per epoch 0.0762989296913147 (for 500 epochs)
Epoch  150  time  0.15281152725219727
Epoch  150  loss  0.17545341497236375 correct 48
Average time per epoch 0.0766045527458191 (for 500 epochs)
Epoch  151  time  0.14713096618652344
Average time per epoch 0.07689881467819214 (for 500 epochs)
Epoch  152  time  0.14548158645629883
Average time per epoch 0.07718977785110473 (for 500 epochs)
Epoch  153  time  0.1409435272216797
Average time per epoch 0.07747166490554809 (for 500 epochs)
Epoch  154  time  0.152815580368042
Average time per epoch 0.07777729606628418 (for 500 epochs)
Epoch  155  time  0.16155576705932617
Average time per epoch 0.07810040760040284 (for 500 epochs)
Epoch  156  time  0.14972901344299316
Average time per epoch 0.07839986562728882 (for 500 epochs)
Epoch  157  time  0.14835357666015625
Average time per epoch 0.07869657278060913 (for 500 epochs)
Epoch  158  time  0.15180659294128418
Average time per epoch 0.07900018596649169 (for 500 epochs)
Epoch  159  time  0.1466522216796875
Average time per epoch 0.07929349040985108 (for 500 epochs)
Epoch  160  time  0.15424442291259766
Epoch  160  loss  0.8448422960337292 correct 48
Average time per epoch 0.07960197925567628 (for 500 epochs)
Epoch  161  time  0.15704822540283203
Average time per epoch 0.07991607570648193 (for 500 epochs)
Epoch  162  time  0.156174898147583
Average time per epoch 0.0802284255027771 (for 500 epochs)
Epoch  163  time  0.14285755157470703
Average time per epoch 0.08051414060592652 (for 500 epochs)
Epoch  164  time  0.15645170211791992
Average time per epoch 0.08082704401016236 (for 500 epochs)
Epoch  165  time  0.1460423469543457
Average time per epoch 0.08111912870407105 (for 500 epochs)
Epoch  166  time  0.15206050872802734
Average time per epoch 0.0814232497215271 (for 500 epochs)
Epoch  167  time  0.1438915729522705
Average time per epoch 0.08171103286743164 (for 500 epochs)
Epoch  168  time  0.16165423393249512
Average time per epoch 0.08203434133529663 (for 500 epochs)
Epoch  169  time  0.1495990753173828
Average time per epoch 0.08233353948593139 (for 500 epochs)
Epoch  170  time  0.14676880836486816
Epoch  170  loss  1.4814163515360266 correct 48
Average time per epoch 0.08262707710266114 (for 500 epochs)
Epoch  171  time  0.15021586418151855
Average time per epoch 0.08292750883102416 (for 500 epochs)
Epoch  172  time  0.1469721794128418
Average time per epoch 0.08322145318984986 (for 500 epochs)
Epoch  173  time  0.14359593391418457
Average time per epoch 0.08350864505767822 (for 500 epochs)
Epoch  174  time  0.15161609649658203
Average time per epoch 0.0838118772506714 (for 500 epochs)
Epoch  175  time  0.16338515281677246
Average time per epoch 0.08413864755630493 (for 500 epochs)
Epoch  176  time  0.14530324935913086
Average time per epoch 0.08442925405502319 (for 500 epochs)
Epoch  177  time  0.15172362327575684
Average time per epoch 0.0847327013015747 (for 500 epochs)
Epoch  178  time  0.14713597297668457
Average time per epoch 0.08502697324752807 (for 500 epochs)
Epoch  179  time  0.14743375778198242
Average time per epoch 0.08532184076309204 (for 500 epochs)
Epoch  180  time  0.15031695365905762
Epoch  180  loss  0.6683824837529517 correct 49
Average time per epoch 0.08562247467041016 (for 500 epochs)
Epoch  181  time  0.14733386039733887
Average time per epoch 0.08591714239120483 (for 500 epochs)
Epoch  182  time  0.16820168495178223
Average time per epoch 0.0862535457611084 (for 500 epochs)
Epoch  183  time  0.14584684371948242
Average time per epoch 0.08654523944854736 (for 500 epochs)
Epoch  184  time  0.15452933311462402
Average time per epoch 0.08685429811477662 (for 500 epochs)
Epoch  185  time  0.15159845352172852
Average time per epoch 0.08715749502182008 (for 500 epochs)
Epoch  186  time  0.15259385108947754
Average time per epoch 0.08746268272399903 (for 500 epochs)
Epoch  187  time  0.14521026611328125
Average time per epoch 0.08775310325622558 (for 500 epochs)
Epoch  188  time  0.15933918952941895
Average time per epoch 0.08807178163528442 (for 500 epochs)
Epoch  189  time  0.14905071258544922
Average time per epoch 0.08836988306045532 (for 500 epochs)
Epoch  190  time  0.1499161720275879
Epoch  190  loss  1.457307635936904 correct 49
Average time per epoch 0.0886697154045105 (for 500 epochs)
Epoch  191  time  0.14098072052001953
Average time per epoch 0.08895167684555054 (for 500 epochs)
Epoch  192  time  0.14334344863891602
Average time per epoch 0.08923836374282837 (for 500 epochs)
Epoch  193  time  0.142791748046875
Average time per epoch 0.08952394723892212 (for 500 epochs)
Epoch  194  time  0.1566925048828125
Average time per epoch 0.08983733224868774 (for 500 epochs)
Epoch  195  time  0.1666567325592041
Average time per epoch 0.09017064571380615 (for 500 epochs)
Epoch  196  time  0.15379571914672852
Average time per epoch 0.09047823715209961 (for 500 epochs)
Epoch  197  time  0.15661001205444336
Average time per epoch 0.0907914571762085 (for 500 epochs)
Epoch  198  time  0.15326547622680664
Average time per epoch 0.09109798812866211 (for 500 epochs)
Epoch  199  time  0.16340112686157227
Average time per epoch 0.09142479038238525 (for 500 epochs)
Epoch  200  time  0.14841604232788086
Epoch  200  loss  0.4799529612354036 correct 49
Average time per epoch 0.09172162246704102 (for 500 epochs)
Epoch  201  time  0.15001678466796875
Average time per epoch 0.09202165603637695 (for 500 epochs)
Epoch  202  time  0.1633443832397461
Average time per epoch 0.09234834480285645 (for 500 epochs)
Epoch  203  time  0.14366483688354492
Average time per epoch 0.09263567447662353 (for 500 epochs)
Epoch  204  time  0.14374470710754395
Average time per epoch 0.09292316389083863 (for 500 epochs)
Epoch  205  time  0.14220523834228516
Average time per epoch 0.0932075743675232 (for 500 epochs)
Epoch  206  time  0.144911527633667
Average time per epoch 0.09349739742279052 (for 500 epochs)
Epoch  207  time  0.140380859375
Average time per epoch 0.09377815914154053 (for 500 epochs)
Epoch  208  time  0.15219950675964355
Average time per epoch 0.09408255815505981 (for 500 epochs)
Epoch  209  time  0.15819668769836426
Average time per epoch 0.09439895153045655 (for 500 epochs)
Epoch  210  time  0.15232419967651367
Epoch  210  loss  1.0424422563044962 correct 48
Average time per epoch 0.09470359992980958 (for 500 epochs)
Epoch  211  time  0.15339064598083496
Average time per epoch 0.09501038122177125 (for 500 epochs)
Epoch  212  time  0.15859723091125488
Average time per epoch 0.09532757568359375 (for 500 epochs)
Epoch  213  time  0.14427804946899414
Average time per epoch 0.09561613178253174 (for 500 epochs)
Epoch  214  time  0.14792585372924805
Average time per epoch 0.09591198348999024 (for 500 epochs)
Epoch  215  time  0.1671898365020752
Average time per epoch 0.09624636316299438 (for 500 epochs)
Epoch  216  time  0.1536712646484375
Average time per epoch 0.09655370569229126 (for 500 epochs)
Epoch  217  time  0.1460733413696289
Average time per epoch 0.09684585237503052 (for 500 epochs)
Epoch  218  time  0.1528332233428955
Average time per epoch 0.09715151882171631 (for 500 epochs)
Epoch  219  time  0.14551830291748047
Average time per epoch 0.09744255542755127 (for 500 epochs)
Epoch  220  time  0.15356087684631348
Epoch  220  loss  0.23064599085105875 correct 49
Average time per epoch 0.0977496771812439 (for 500 epochs)
Epoch  221  time  0.14312291145324707
Average time per epoch 0.09803592300415039 (for 500 epochs)
Epoch  222  time  0.1653423309326172
Average time per epoch 0.09836660766601563 (for 500 epochs)
Epoch  223  time  0.14940905570983887
Average time per epoch 0.0986654257774353 (for 500 epochs)
Epoch  224  time  0.15579462051391602
Average time per epoch 0.09897701501846314 (for 500 epochs)
Epoch  225  time  0.15247654914855957
Average time per epoch 0.09928196811676025 (for 500 epochs)
Epoch  226  time  0.14944958686828613
Average time per epoch 0.09958086729049682 (for 500 epochs)
Epoch  227  time  0.15577197074890137
Average time per epoch 0.09989241123199463 (for 500 epochs)
Epoch  228  time  0.15496039390563965
Average time per epoch 0.10020233201980591 (for 500 epochs)
Epoch  229  time  0.16716480255126953
Average time per epoch 0.10053666162490844 (for 500 epochs)
Epoch  230  time  0.14684438705444336
Epoch  230  loss  0.028506229431390147 correct 48
Average time per epoch 0.10083035039901733 (for 500 epochs)
Epoch  231  time  0.15192580223083496
Average time per epoch 0.101134202003479 (for 500 epochs)
Epoch  232  time  0.14732766151428223
Average time per epoch 0.10142885732650757 (for 500 epochs)
Epoch  233  time  0.14404940605163574
Average time per epoch 0.10171695613861084 (for 500 epochs)
Epoch  234  time  0.16599035263061523
Average time per epoch 0.10204893684387208 (for 500 epochs)
Epoch  235  time  0.2522718906402588
Average time per epoch 0.10255348062515258 (for 500 epochs)
Epoch  236  time  0.724571943283081
Average time per epoch 0.10400262451171875 (for 500 epochs)
Epoch  237  time  0.5155055522918701
Average time per epoch 0.1050336356163025 (for 500 epochs)
Epoch  238  time  2.90972638130188
Average time per epoch 0.11085308837890626 (for 500 epochs)
Epoch  239  time  2.086855173110962
Average time per epoch 0.11502679872512818 (for 500 epochs)
Epoch  240  time  0.9316685199737549
Epoch  240  loss  1.5048978356214051 correct 48
Average time per epoch 0.11689013576507569 (for 500 epochs)
Epoch  241  time  0.8144416809082031
Average time per epoch 0.11851901912689208 (for 500 epochs)
Epoch  242  time  2.782388210296631
Average time per epoch 0.12408379554748535 (for 500 epochs)
Epoch  243  time  0.9581003189086914
Average time per epoch 0.12599999618530272 (for 500 epochs)
Epoch  244  time  0.1539459228515625
Average time per epoch 0.12630788803100587 (for 500 epochs)
Epoch  245  time  0.15176820755004883
Average time per epoch 0.12661142444610596 (for 500 epochs)
Epoch  246  time  0.1435251235961914
Average time per epoch 0.12689847469329835 (for 500 epochs)
Epoch  247  time  0.1712186336517334
Average time per epoch 0.1272409119606018 (for 500 epochs)
Epoch  248  time  0.14569878578186035
Average time per epoch 0.12753230953216552 (for 500 epochs)
Epoch  249  time  0.15374422073364258
Average time per epoch 0.12783979797363282 (for 500 epochs)
Epoch  250  time  0.1416013240814209
Epoch  250  loss  1.1331973611768251 correct 48
Average time per epoch 0.12812300062179566 (for 500 epochs)
Epoch  251  time  0.15434527397155762
Average time per epoch 0.12843169116973877 (for 500 epochs)
Epoch  252  time  0.1563282012939453
Average time per epoch 0.12874434757232667 (for 500 epochs)
Epoch  253  time  0.175215482711792
Average time per epoch 0.12909477853775025 (for 500 epochs)
Epoch  254  time  0.14508914947509766
Average time per epoch 0.12938495683670043 (for 500 epochs)
Epoch  255  time  0.1472935676574707
Average time per epoch 0.1296795439720154 (for 500 epochs)
Epoch  256  time  0.15657329559326172
Average time per epoch 0.1299926905632019 (for 500 epochs)
Epoch  257  time  0.16273188591003418
Average time per epoch 0.13031815433502197 (for 500 epochs)
Epoch  258  time  0.1466083526611328
Average time per epoch 0.13061137104034423 (for 500 epochs)
Epoch  259  time  0.1522223949432373
Average time per epoch 0.1309158158302307 (for 500 epochs)
Epoch  260  time  0.1573038101196289
Epoch  260  loss  0.5910764282853763 correct 50
Average time per epoch 0.13123042345046998 (for 500 epochs)
Epoch  261  time  0.1478290557861328
Average time per epoch 0.13152608156204224 (for 500 epochs)
Epoch  262  time  0.1473255157470703
Average time per epoch 0.13182073259353638 (for 500 epochs)
Epoch  263  time  0.15279793739318848
Average time per epoch 0.13212632846832276 (for 500 epochs)
Epoch  264  time  0.14918828010559082
Average time per epoch 0.13242470502853393 (for 500 epochs)
Epoch  265  time  0.1483631134033203
Average time per epoch 0.13272143125534058 (for 500 epochs)
Epoch  266  time  0.14189577102661133
Average time per epoch 0.1330052227973938 (for 500 epochs)
Epoch  267  time  0.1620190143585205
Average time per epoch 0.13332926082611085 (for 500 epochs)
Epoch  268  time  0.14799809455871582
Average time per epoch 0.13362525701522826 (for 500 epochs)
Epoch  269  time  0.1502668857574463
Average time per epoch 0.13392579078674316 (for 500 epochs)
Epoch  270  time  0.1456007957458496
Epoch  270  loss  1.3544933731068012 correct 48
Average time per epoch 0.13421699237823487 (for 500 epochs)
Epoch  271  time  0.1544022560119629
Average time per epoch 0.1345257968902588 (for 500 epochs)
Epoch  272  time  0.1434330940246582
Average time per epoch 0.1348126630783081 (for 500 epochs)
Epoch  273  time  0.16351532936096191
Average time per epoch 0.13513969373703003 (for 500 epochs)
Epoch  274  time  0.14522838592529297
Average time per epoch 0.1354301505088806 (for 500 epochs)
Epoch  275  time  0.1466376781463623
Average time per epoch 0.13572342586517333 (for 500 epochs)
Epoch  276  time  0.15137052536010742
Average time per epoch 0.13602616691589356 (for 500 epochs)
Epoch  277  time  0.15297460556030273
Average time per epoch 0.13633211612701415 (for 500 epochs)
Epoch  278  time  0.14703583717346191
Average time per epoch 0.13662618780136107 (for 500 epochs)
Epoch  279  time  0.17363381385803223
Average time per epoch 0.13697345542907716 (for 500 epochs)
Epoch  280  time  0.16212844848632812
Epoch  280  loss  1.1011235095349927 correct 48
Average time per epoch 0.13729771232604981 (for 500 epochs)
Epoch  281  time  0.1419389247894287
Average time per epoch 0.13758159017562865 (for 500 epochs)
Epoch  282  time  0.14998674392700195
Average time per epoch 0.13788156366348267 (for 500 epochs)
Epoch  283  time  0.14469408988952637
Average time per epoch 0.13817095184326172 (for 500 epochs)
Epoch  284  time  0.1498405933380127
Average time per epoch 0.13847063302993776 (for 500 epochs)
Epoch  285  time  0.14730095863342285
Average time per epoch 0.13876523494720458 (for 500 epochs)
Epoch  286  time  0.1508166790008545
Average time per epoch 0.1390668683052063 (for 500 epochs)
Epoch  287  time  0.1679079532623291
Average time per epoch 0.13940268421173097 (for 500 epochs)
Epoch  288  time  0.1668996810913086
Average time per epoch 0.13973648357391358 (for 500 epochs)
Epoch  289  time  0.160400390625
Average time per epoch 0.14005728435516357 (for 500 epochs)
Epoch  290  time  0.17876291275024414
Epoch  290  loss  1.0712105227600355 correct 49
Average time per epoch 0.14041481018066407 (for 500 epochs)
Epoch  291  time  0.14310097694396973
Average time per epoch 0.140701012134552 (for 500 epochs)
Epoch  292  time  0.15493011474609375
Average time per epoch 0.1410108723640442 (for 500 epochs)
Epoch  293  time  0.158538818359375
Average time per epoch 0.14132795000076295 (for 500 epochs)
Epoch  294  time  0.1487414836883545
Average time per epoch 0.14162543296813965 (for 500 epochs)
Epoch  295  time  0.15184473991394043
Average time per epoch 0.14192912244796754 (for 500 epochs)
Epoch  296  time  0.15514826774597168
Average time per epoch 0.14223941898345946 (for 500 epochs)
Epoch  297  time  0.14731550216674805
Average time per epoch 0.14253404998779298 (for 500 epochs)
Epoch  298  time  0.14928627014160156
Average time per epoch 0.14283262252807616 (for 500 epochs)
Epoch  299  time  0.15845203399658203
Average time per epoch 0.14314952659606933 (for 500 epochs)
Epoch  300  time  0.18919014930725098
Epoch  300  loss  0.37353635670589974 correct 48
Average time per epoch 0.14352790689468384 (for 500 epochs)
Epoch  301  time  0.15266966819763184
Average time per epoch 0.1438332462310791 (for 500 epochs)
Epoch  302  time  0.18441081047058105
Average time per epoch 0.14420206785202028 (for 500 epochs)
Epoch  303  time  0.15537285804748535
Average time per epoch 0.14451281356811524 (for 500 epochs)
Epoch  304  time  0.14724349975585938
Average time per epoch 0.14480730056762695 (for 500 epochs)
Epoch  305  time  0.14641690254211426
Average time per epoch 0.1451001343727112 (for 500 epochs)
Epoch  306  time  0.16591191291809082
Average time per epoch 0.14543195819854737 (for 500 epochs)
Epoch  307  time  0.15177273750305176
Average time per epoch 0.14573550367355348 (for 500 epochs)
Epoch  308  time  0.15047812461853027
Average time per epoch 0.14603645992279052 (for 500 epochs)
Epoch  309  time  0.1453876495361328
Average time per epoch 0.1463272352218628 (for 500 epochs)
Epoch  310  time  0.14551115036010742
Epoch  310  loss  3.3626226503119065 correct 47
Average time per epoch 0.146618257522583 (for 500 epochs)
Epoch  311  time  0.14289045333862305
Average time per epoch 0.14690403842926025 (for 500 epochs)
Epoch  312  time  0.15086770057678223
Average time per epoch 0.1472057738304138 (for 500 epochs)
Epoch  313  time  0.1680004596710205
Average time per epoch 0.14754177474975586 (for 500 epochs)
Epoch  314  time  0.15211129188537598
Average time per epoch 0.1478459973335266 (for 500 epochs)
Epoch  315  time  0.15080499649047852
Average time per epoch 0.14814760732650756 (for 500 epochs)
Epoch  316  time  0.14858198165893555
Average time per epoch 0.14844477128982544 (for 500 epochs)
Epoch  317  time  0.14545655250549316
Average time per epoch 0.14873568439483642 (for 500 epochs)
Epoch  318  time  0.1440894603729248
Average time per epoch 0.1490238633155823 (for 500 epochs)
Epoch  319  time  0.15722441673278809
Average time per epoch 0.14933831214904786 (for 500 epochs)
Epoch  320  time  0.15941715240478516
Epoch  320  loss  0.7880298078988515 correct 48
Average time per epoch 0.14965714645385741 (for 500 epochs)
Epoch  321  time  0.16283392906188965
Average time per epoch 0.1499828143119812 (for 500 epochs)
Epoch  322  time  0.1730666160583496
Average time per epoch 0.1503289475440979 (for 500 epochs)
Epoch  323  time  0.1546471118927002
Average time per epoch 0.1506382417678833 (for 500 epochs)
Epoch  324  time  0.1472940444946289
Average time per epoch 0.15093282985687256 (for 500 epochs)
Epoch  325  time  0.1583714485168457
Average time per epoch 0.15124957275390624 (for 500 epochs)
Epoch  326  time  0.17064547538757324
Average time per epoch 0.1515908637046814 (for 500 epochs)
Epoch  327  time  0.14636802673339844
Average time per epoch 0.1518835997581482 (for 500 epochs)
Epoch  328  time  0.1627662181854248
Average time per epoch 0.15220913219451904 (for 500 epochs)
Epoch  329  time  0.14996695518493652
Average time per epoch 0.15250906610488893 (for 500 epochs)
Epoch  330  time  0.15100932121276855
Epoch  330  loss  0.15556478529744003 correct 49
Average time per epoch 0.15281108474731445 (for 500 epochs)
Epoch  331  time  0.14950346946716309
Average time per epoch 0.15311009168624878 (for 500 epochs)
Epoch  332  time  0.2283005714416504
Average time per epoch 0.15356669282913207 (for 500 epochs)
Epoch  333  time  0.1480100154876709
Average time per epoch 0.15386271286010741 (for 500 epochs)
Epoch  334  time  0.15179038047790527
Average time per epoch 0.15416629362106324 (for 500 epochs)
Epoch  335  time  0.15105676651000977
Average time per epoch 0.15446840715408325 (for 500 epochs)
Epoch  336  time  0.14937496185302734
Average time per epoch 0.15476715707778932 (for 500 epochs)
Epoch  337  time  0.14182376861572266
Average time per epoch 0.15505080461502074 (for 500 epochs)
Epoch  338  time  0.1499185562133789
Average time per epoch 0.1553506417274475 (for 500 epochs)
Epoch  339  time  0.16179847717285156
Average time per epoch 0.15567423868179323 (for 500 epochs)
Epoch  340  time  0.15302491188049316
Epoch  340  loss  2.197710658342701 correct 48
Average time per epoch 0.1559802885055542 (for 500 epochs)
Epoch  341  time  0.14781451225280762
Average time per epoch 0.15627591753005982 (for 500 epochs)
Epoch  342  time  0.1544029712677002
Average time per epoch 0.15658472347259522 (for 500 epochs)
Epoch  343  time  0.14557266235351562
Average time per epoch 0.15687586879730225 (for 500 epochs)
Epoch  344  time  0.14711284637451172
Average time per epoch 0.15717009449005126 (for 500 epochs)
Epoch  345  time  0.18691039085388184
Average time per epoch 0.15754391527175904 (for 500 epochs)
Epoch  346  time  0.14986777305603027
Average time per epoch 0.1578436508178711 (for 500 epochs)
Epoch  347  time  0.1452336311340332
Average time per epoch 0.15813411808013916 (for 500 epochs)
Epoch  348  time  0.14457988739013672
Average time per epoch 0.15842327785491944 (for 500 epochs)
Epoch  349  time  0.15202832221984863
Average time per epoch 0.15872733449935914 (for 500 epochs)
Epoch  350  time  0.14298129081726074
Epoch  350  loss  0.8002262603170671 correct 50
Average time per epoch 0.15901329708099365 (for 500 epochs)
Epoch  351  time  0.15242910385131836
Average time per epoch 0.15931815528869628 (for 500 epochs)
Epoch  352  time  0.1634223461151123
Average time per epoch 0.15964499998092652 (for 500 epochs)
Epoch  353  time  0.16928505897521973
Average time per epoch 0.15998357009887695 (for 500 epochs)
Epoch  354  time  0.1578226089477539
Average time per epoch 0.16029921531677246 (for 500 epochs)
Epoch  355  time  0.15044260025024414
Average time per epoch 0.16060010051727294 (for 500 epochs)
Epoch  356  time  0.14188480377197266
Average time per epoch 0.1608838701248169 (for 500 epochs)
Epoch  357  time  0.14944124221801758
Average time per epoch 0.16118275260925294 (for 500 epochs)
Epoch  358  time  0.1515955924987793
Average time per epoch 0.1614859437942505 (for 500 epochs)
Epoch  359  time  0.16727828979492188
Average time per epoch 0.16182050037384033 (for 500 epochs)
Epoch  360  time  0.15379667282104492
Epoch  360  loss  0.18742310236288687 correct 49
Average time per epoch 0.16212809371948242 (for 500 epochs)
Epoch  361  time  0.14999818801879883
Average time per epoch 0.16242809009552 (for 500 epochs)
Epoch  362  time  0.14175152778625488
Average time per epoch 0.16271159315109254 (for 500 epochs)
Epoch  363  time  0.14484477043151855
Average time per epoch 0.16300128269195557 (for 500 epochs)
Epoch  364  time  0.1526198387145996
Average time per epoch 0.16330652236938475 (for 500 epochs)
Epoch  365  time  0.15474176406860352
Average time per epoch 0.16361600589752198 (for 500 epochs)
Epoch  366  time  0.14580869674682617
Average time per epoch 0.16390762329101563 (for 500 epochs)
Epoch  367  time  0.14772343635559082
Average time per epoch 0.16420307016372682 (for 500 epochs)
Epoch  368  time  0.15174245834350586
Average time per epoch 0.1645065550804138 (for 500 epochs)
Epoch  369  time  0.14962339401245117
Average time per epoch 0.1648058018684387 (for 500 epochs)
Epoch  370  time  0.1530609130859375
Epoch  370  loss  0.18600689929506958 correct 50
Average time per epoch 0.16511192369461059 (for 500 epochs)
Epoch  371  time  0.14827752113342285
Average time per epoch 0.16540847873687745 (for 500 epochs)
Epoch  372  time  0.16925859451293945
Average time per epoch 0.1657469959259033 (for 500 epochs)
Epoch  373  time  0.14811158180236816
Average time per epoch 0.16604321908950806 (for 500 epochs)
Epoch  374  time  0.1487746238708496
Average time per epoch 0.16634076833724976 (for 500 epochs)
Epoch  375  time  0.1464078426361084
Average time per epoch 0.16663358402252199 (for 500 epochs)
Epoch  376  time  0.1449446678161621
Average time per epoch 0.1669234733581543 (for 500 epochs)
Epoch  377  time  0.1492307186126709
Average time per epoch 0.16722193479537964 (for 500 epochs)
Epoch  378  time  0.1456012725830078
Average time per epoch 0.16751313734054565 (for 500 epochs)
Epoch  379  time  0.16727089881896973
Average time per epoch 0.1678476791381836 (for 500 epochs)
Epoch  380  time  0.14559650421142578
Epoch  380  loss  1.277434012918472 correct 48
Average time per epoch 0.16813887214660644 (for 500 epochs)
Epoch  381  time  0.15364956855773926
Average time per epoch 0.1684461712837219 (for 500 epochs)
Epoch  382  time  0.144911527633667
Average time per epoch 0.16873599433898925 (for 500 epochs)
Epoch  383  time  0.1549205780029297
Average time per epoch 0.1690458354949951 (for 500 epochs)
Epoch  384  time  0.14434409141540527
Average time per epoch 0.16933452367782592 (for 500 epochs)
Epoch  385  time  0.15009212493896484
Average time per epoch 0.16963470792770385 (for 500 epochs)
Epoch  386  time  0.1631307601928711
Average time per epoch 0.1699609694480896 (for 500 epochs)
Epoch  387  time  0.14565634727478027
Average time per epoch 0.17025228214263916 (for 500 epochs)
Epoch  388  time  0.1589503288269043
Average time per epoch 0.17057018280029296 (for 500 epochs)
Epoch  389  time  0.1463172435760498
Average time per epoch 0.17086281728744507 (for 500 epochs)
Epoch  390  time  0.14893126487731934
Epoch  390  loss  0.1563298012472594 correct 48
Average time per epoch 0.1711606798171997 (for 500 epochs)
Epoch  391  time  0.1515977382659912
Average time per epoch 0.1714638752937317 (for 500 epochs)
Epoch  392  time  0.1607041358947754
Average time per epoch 0.17178528356552125 (for 500 epochs)
Epoch  393  time  0.1575484275817871
Average time per epoch 0.17210038042068482 (for 500 epochs)
Epoch  394  time  0.14474916458129883
Average time per epoch 0.1723898787498474 (for 500 epochs)
Epoch  395  time  0.15331792831420898
Average time per epoch 0.17269651460647584 (for 500 epochs)
Epoch  396  time  0.14679932594299316
Average time per epoch 0.1729901132583618 (for 500 epochs)
Epoch  397  time  0.15004682540893555
Average time per epoch 0.17329020690917968 (for 500 epochs)
Epoch  398  time  0.14683294296264648
Average time per epoch 0.17358387279510498 (for 500 epochs)
Epoch  399  time  0.1625983715057373
Average time per epoch 0.17390906953811647 (for 500 epochs)
Epoch  400  time  0.14842820167541504
Epoch  400  loss  1.0902388360104083 correct 50
Average time per epoch 0.17420592594146728 (for 500 epochs)
Epoch  401  time  0.15154504776000977
Average time per epoch 0.1745090160369873 (for 500 epochs)
Epoch  402  time  0.1524815559387207
Average time per epoch 0.17481397914886473 (for 500 epochs)
Epoch  403  time  0.15275335311889648
Average time per epoch 0.17511948585510254 (for 500 epochs)
Epoch  404  time  0.1530444622039795
Average time per epoch 0.17542557477951048 (for 500 epochs)
Epoch  405  time  0.14816832542419434
Average time per epoch 0.1757219114303589 (for 500 epochs)
Epoch  406  time  0.16246414184570312
Average time per epoch 0.1760468397140503 (for 500 epochs)
Epoch  407  time  0.1530148983001709
Average time per epoch 0.17635286951065063 (for 500 epochs)
Epoch  408  time  0.14407706260681152
Average time per epoch 0.17664102363586426 (for 500 epochs)
Epoch  409  time  0.1529092788696289
Average time per epoch 0.17694684219360352 (for 500 epochs)
Epoch  410  time  0.14960145950317383
Epoch  410  loss  0.01377308606164907 correct 50
Average time per epoch 0.17724604511260986 (for 500 epochs)
Epoch  411  time  0.14884138107299805
Average time per epoch 0.17754372787475586 (for 500 epochs)
Epoch  412  time  0.18039226531982422
Average time per epoch 0.1779045124053955 (for 500 epochs)
Epoch  413  time  0.1522364616394043
Average time per epoch 0.17820898532867432 (for 500 epochs)
Epoch  414  time  0.143629789352417
Average time per epoch 0.17849624490737914 (for 500 epochs)
Epoch  415  time  0.14881467819213867
Average time per epoch 0.17879387426376342 (for 500 epochs)
Epoch  416  time  0.14858341217041016
Average time per epoch 0.17909104108810425 (for 500 epochs)
Epoch  417  time  0.16969799995422363
Average time per epoch 0.1794304370880127 (for 500 epochs)
Epoch  418  time  0.1481633186340332
Average time per epoch 0.17972676372528076 (for 500 epochs)
Epoch  419  time  0.16315698623657227
Average time per epoch 0.18005307769775392 (for 500 epochs)
Epoch  420  time  0.14220023155212402
Epoch  420  loss  0.4391648697234418 correct 50
Average time per epoch 0.18033747816085816 (for 500 epochs)
Epoch  421  time  0.14566898345947266
Average time per epoch 0.1806288161277771 (for 500 epochs)
Epoch  422  time  0.14434456825256348
Average time per epoch 0.1809175052642822 (for 500 epochs)
Epoch  423  time  0.1507267951965332
Average time per epoch 0.1812189588546753 (for 500 epochs)
Epoch  424  time  0.15080642700195312
Average time per epoch 0.1815205717086792 (for 500 epochs)
Epoch  425  time  0.150054931640625
Average time per epoch 0.18182068157196046 (for 500 epochs)
Epoch  426  time  0.17000627517700195
Average time per epoch 0.18216069412231445 (for 500 epochs)
Epoch  427  time  0.1467123031616211
Average time per epoch 0.1824541187286377 (for 500 epochs)
Epoch  428  time  0.14120006561279297
Average time per epoch 0.18273651885986328 (for 500 epochs)
Epoch  429  time  0.1513974666595459
Average time per epoch 0.18303931379318236 (for 500 epochs)
Epoch  430  time  0.1421644687652588
Epoch  430  loss  0.26009090756234465 correct 48
Average time per epoch 0.1833236427307129 (for 500 epochs)
Epoch  431  time  0.14951539039611816
Average time per epoch 0.18362267351150513 (for 500 epochs)
Epoch  432  time  0.15769052505493164
Average time per epoch 0.183938054561615 (for 500 epochs)
Epoch  433  time  0.15877127647399902
Average time per epoch 0.184255597114563 (for 500 epochs)
Epoch  434  time  0.16323471069335938
Average time per epoch 0.18458206653594972 (for 500 epochs)
Epoch  435  time  0.15647387504577637
Average time per epoch 0.18489501428604127 (for 500 epochs)
Epoch  436  time  0.16006684303283691
Average time per epoch 0.18521514797210695 (for 500 epochs)
Epoch  437  time  0.1558091640472412
Average time per epoch 0.18552676630020143 (for 500 epochs)
Epoch  438  time  0.14381957054138184
Average time per epoch 0.1858144054412842 (for 500 epochs)
Epoch  439  time  0.16255712509155273
Average time per epoch 0.18613951969146728 (for 500 epochs)
Epoch  440  time  0.13948822021484375
Epoch  440  loss  1.7839796139653197 correct 48
Average time per epoch 0.18641849613189698 (for 500 epochs)
Epoch  441  time  0.15051889419555664
Average time per epoch 0.1867195339202881 (for 500 epochs)
Epoch  442  time  0.14008569717407227
Average time per epoch 0.18699970531463622 (for 500 epochs)
Epoch  443  time  0.14733195304870605
Average time per epoch 0.18729436922073364 (for 500 epochs)
Epoch  444  time  0.14686131477355957
Average time per epoch 0.18758809185028077 (for 500 epochs)
Epoch  445  time  0.1451869010925293
Average time per epoch 0.18787846565246583 (for 500 epochs)
Epoch  446  time  0.16203880310058594
Average time per epoch 0.188202543258667 (for 500 epochs)
Epoch  447  time  0.14268088340759277
Average time per epoch 0.18848790502548218 (for 500 epochs)
Epoch  448  time  0.15594267845153809
Average time per epoch 0.18879979038238526 (for 500 epochs)
Epoch  449  time  0.16220450401306152
Average time per epoch 0.1891241993904114 (for 500 epochs)
Epoch  450  time  0.1623694896697998
Epoch  450  loss  0.003423381455455608 correct 50
Average time per epoch 0.18944893836975096 (for 500 epochs)
Epoch  451  time  0.15374040603637695
Average time per epoch 0.18975641918182373 (for 500 epochs)
Epoch  452  time  0.15788698196411133
Average time per epoch 0.19007219314575197 (for 500 epochs)
Epoch  453  time  0.1493074893951416
Average time per epoch 0.19037080812454224 (for 500 epochs)
Epoch  454  time  0.15417218208312988
Average time per epoch 0.1906791524887085 (for 500 epochs)
Epoch  455  time  0.14189958572387695
Average time per epoch 0.19096295166015625 (for 500 epochs)
Epoch  456  time  0.14882636070251465
Average time per epoch 0.19126060438156128 (for 500 epochs)
Epoch  457  time  0.14150643348693848
Average time per epoch 0.19154361724853516 (for 500 epochs)
Epoch  458  time  0.15266990661621094
Average time per epoch 0.19184895706176758 (for 500 epochs)
Epoch  459  time  0.15525150299072266
Average time per epoch 0.19215946006774903 (for 500 epochs)
Epoch  460  time  0.1475210189819336
Epoch  460  loss  1.2461378950247028 correct 48
Average time per epoch 0.19245450210571288 (for 500 epochs)
Epoch  461  time  0.14600706100463867
Average time per epoch 0.19274651622772218 (for 500 epochs)
Epoch  462  time  0.1472330093383789
Average time per epoch 0.19304098224639893 (for 500 epochs)
Epoch  463  time  0.1450517177581787
Average time per epoch 0.1933310856819153 (for 500 epochs)
Epoch  464  time  0.1462385654449463
Average time per epoch 0.19362356281280518 (for 500 epochs)
Epoch  465  time  0.1440718173980713
Average time per epoch 0.1939117064476013 (for 500 epochs)
Epoch  466  time  0.16143298149108887
Average time per epoch 0.1942345724105835 (for 500 epochs)
Epoch  467  time  0.14307904243469238
Average time per epoch 0.19452073049545288 (for 500 epochs)
Epoch  468  time  0.15381312370300293
Average time per epoch 0.1948283567428589 (for 500 epochs)
Epoch  469  time  0.14377593994140625
Average time per epoch 0.1951159086227417 (for 500 epochs)
Epoch  470  time  0.1557903289794922
Epoch  470  loss  0.08897860654107799 correct 50
Average time per epoch 0.1954274892807007 (for 500 epochs)
Epoch  471  time  0.1441049575805664
Average time per epoch 0.19571569919586182 (for 500 epochs)
Epoch  472  time  0.1452946662902832
Average time per epoch 0.19600628852844237 (for 500 epochs)
Epoch  473  time  0.15267372131347656
Average time per epoch 0.19631163597106935 (for 500 epochs)
Epoch  474  time  0.14277434349060059
Average time per epoch 0.19659718465805054 (for 500 epochs)
Epoch  475  time  0.14073705673217773
Average time per epoch 0.1968786587715149 (for 500 epochs)
Epoch  476  time  0.14745712280273438
Average time per epoch 0.19717357301712035 (for 500 epochs)
Epoch  477  time  0.14210176467895508
Average time per epoch 0.19745777654647828 (for 500 epochs)
Epoch  478  time  0.15469694137573242
Average time per epoch 0.19776717042922973 (for 500 epochs)
Epoch  479  time  0.1449909210205078
Average time per epoch 0.19805715227127074 (for 500 epochs)
Epoch  480  time  0.18006300926208496
Epoch  480  loss  0.5424859705126945 correct 49
Average time per epoch 0.1984172782897949 (for 500 epochs)
Epoch  481  time  0.1386275291442871
Average time per epoch 0.19869453334808349 (for 500 epochs)
Epoch  482  time  0.15129709243774414
Average time per epoch 0.19899712753295898 (for 500 epochs)
Epoch  483  time  0.14071011543273926
Average time per epoch 0.19927854776382448 (for 500 epochs)
Epoch  484  time  0.1513361930847168
Average time per epoch 0.1995812201499939 (for 500 epochs)
Epoch  485  time  0.14191508293151855
Average time per epoch 0.19986505031585694 (for 500 epochs)
Epoch  486  time  0.15692567825317383
Average time per epoch 0.20017890167236327 (for 500 epochs)
Epoch  487  time  0.1381525993347168
Average time per epoch 0.20045520687103272 (for 500 epochs)
Epoch  488  time  0.1428062915802002
Average time per epoch 0.20074081945419311 (for 500 epochs)
Epoch  489  time  0.14451932907104492
Average time per epoch 0.2010298581123352 (for 500 epochs)
Epoch  490  time  0.14752745628356934
Epoch  490  loss  0.000440132495239682 correct 47
Average time per epoch 0.20132491302490235 (for 500 epochs)
Epoch  491  time  0.14286279678344727
Average time per epoch 0.20161063861846923 (for 500 epochs)
Epoch  492  time  0.15079450607299805
Average time per epoch 0.20191222763061523 (for 500 epochs)
Epoch  493  time  0.16171717643737793
Average time per epoch 0.20223566198349 (for 500 epochs)
Epoch  494  time  0.1447458267211914
Average time per epoch 0.20252515363693238 (for 500 epochs)
Epoch  495  time  0.14413142204284668
Average time per epoch 0.20281341648101806 (for 500 epochs)
Epoch  496  time  0.14801430702209473
Average time per epoch 0.20310944509506226 (for 500 epochs)
Epoch  497  time  0.14381933212280273
Average time per epoch 0.20339708375930787 (for 500 epochs)
Epoch  498  time  0.14836573600769043
Average time per epoch 0.20369381523132324 (for 500 epochs)
Epoch  499  time  0.1413419246673584
Average time per epoch 0.20397649908065796 (for 500 epochs)
```
# CPU Split Dataset:
```
Epoch  0  time  15.727854251861572
Epoch  0  loss  6.027463971548986 correct 36
Average time per epoch 0.031455708503723145 (for 500 epochs)
Epoch  1  time  0.1499190330505371
Average time per epoch 0.03175554656982422 (for 500 epochs)
Epoch  2  time  0.15368366241455078
Average time per epoch 0.03206291389465332 (for 500 epochs)
Epoch  3  time  0.39856600761413574
Average time per epoch 0.03286004590988159 (for 500 epochs)
Epoch  4  time  0.5617241859436035
Average time per epoch 0.0339834942817688 (for 500 epochs)
Epoch  5  time  1.4885280132293701
Average time per epoch 0.03696055030822754 (for 500 epochs)
Epoch  6  time  0.9513847827911377
Average time per epoch 0.038863319873809817 (for 500 epochs)
Epoch  7  time  1.0174524784088135
Average time per epoch 0.04089822483062744 (for 500 epochs)
Epoch  8  time  0.9277362823486328
Average time per epoch 0.042753697395324704 (for 500 epochs)
Epoch  9  time  2.5696146488189697
Average time per epoch 0.04789292669296265 (for 500 epochs)
Epoch  10  time  4.0560595989227295
Epoch  10  loss  4.601087524028555 correct 36
Average time per epoch 0.056005045890808104 (for 500 epochs)
Epoch  11  time  0.14863801002502441
Average time per epoch 0.056302321910858155 (for 500 epochs)
Epoch  12  time  0.15429162979125977
Average time per epoch 0.05661090517044067 (for 500 epochs)
Epoch  13  time  0.1469719409942627
Average time per epoch 0.0569048490524292 (for 500 epochs)
Epoch  14  time  0.15286779403686523
Average time per epoch 0.05721058464050293 (for 500 epochs)
Epoch  15  time  0.14436888694763184
Average time per epoch 0.057499322414398195 (for 500 epochs)
Epoch  16  time  0.14982080459594727
Average time per epoch 0.05779896402359009 (for 500 epochs)
Epoch  17  time  0.15507817268371582
Average time per epoch 0.05810912036895752 (for 500 epochs)
Epoch  18  time  0.15276575088500977
Average time per epoch 0.05841465187072754 (for 500 epochs)
Epoch  19  time  0.15502452850341797
Average time per epoch 0.05872470092773437 (for 500 epochs)
Epoch  20  time  0.16000080108642578
Epoch  20  loss  5.008136310173204 correct 40
Average time per epoch 0.05904470252990723 (for 500 epochs)
Epoch  21  time  0.15674138069152832
Average time per epoch 0.059358185291290284 (for 500 epochs)
Epoch  22  time  0.1501026153564453
Average time per epoch 0.059658390522003175 (for 500 epochs)
Epoch  23  time  0.16915392875671387
Average time per epoch 0.0599966983795166 (for 500 epochs)
Epoch  24  time  0.14936327934265137
Average time per epoch 0.060295424938201905 (for 500 epochs)
Epoch  25  time  0.15503334999084473
Average time per epoch 0.060605491638183594 (for 500 epochs)
Epoch  26  time  0.15378046035766602
Average time per epoch 0.06091305255889892 (for 500 epochs)
Epoch  27  time  0.15453076362609863
Average time per epoch 0.06122211408615112 (for 500 epochs)
Epoch  28  time  0.15770578384399414
Average time per epoch 0.06153752565383911 (for 500 epochs)
Epoch  29  time  0.21689796447753906
Average time per epoch 0.06197132158279419 (for 500 epochs)
Epoch  30  time  1.288351058959961
Epoch  30  loss  2.7457127050951455 correct 38
Average time per epoch 0.06454802370071411 (for 500 epochs)
Epoch  31  time  0.9824461936950684
Average time per epoch 0.06651291608810425 (for 500 epochs)
Epoch  32  time  0.4012320041656494
Average time per epoch 0.06731538009643555 (for 500 epochs)
Epoch  33  time  1.4904227256774902
Average time per epoch 0.07029622554779053 (for 500 epochs)
Epoch  34  time  2.072031021118164
Average time per epoch 0.07444028759002685 (for 500 epochs)
Epoch  35  time  1.8030750751495361
Average time per epoch 0.07804643774032592 (for 500 epochs)
Epoch  36  time  1.3738563060760498
Average time per epoch 0.08079415035247803 (for 500 epochs)
Epoch  37  time  0.6258397102355957
Average time per epoch 0.08204582977294922 (for 500 epochs)
Epoch  38  time  1.4266886711120605
Average time per epoch 0.08489920711517333 (for 500 epochs)
Epoch  39  time  0.27005672454833984
Average time per epoch 0.08543932056427002 (for 500 epochs)
Epoch  40  time  0.15106773376464844
Epoch  40  loss  2.181979237034769 correct 43
Average time per epoch 0.08574145603179932 (for 500 epochs)
Epoch  41  time  0.15089654922485352
Average time per epoch 0.08604324913024902 (for 500 epochs)
Epoch  42  time  0.14879441261291504
Average time per epoch 0.08634083795547486 (for 500 epochs)
Epoch  43  time  0.15505409240722656
Average time per epoch 0.0866509461402893 (for 500 epochs)
Epoch  44  time  0.1621389389038086
Average time per epoch 0.08697522401809693 (for 500 epochs)
Epoch  45  time  0.1575479507446289
Average time per epoch 0.08729031991958618 (for 500 epochs)
Epoch  46  time  0.15061593055725098
Average time per epoch 0.08759155178070069 (for 500 epochs)
Epoch  47  time  0.14604401588439941
Average time per epoch 0.08788363981246948 (for 500 epochs)
Epoch  48  time  0.15425920486450195
Average time per epoch 0.08819215822219849 (for 500 epochs)
Epoch  49  time  0.15063738822937012
Average time per epoch 0.08849343299865722 (for 500 epochs)
Epoch  50  time  0.16129827499389648
Epoch  50  loss  2.5150663948154146 correct 49
Average time per epoch 0.08881602954864502 (for 500 epochs)
Epoch  51  time  0.15761685371398926
Average time per epoch 0.089131263256073 (for 500 epochs)
Epoch  52  time  0.14835500717163086
Average time per epoch 0.08942797327041627 (for 500 epochs)
Epoch  53  time  0.15354084968566895
Average time per epoch 0.08973505496978759 (for 500 epochs)
Epoch  54  time  0.14808177947998047
Average time per epoch 0.09003121852874756 (for 500 epochs)
Epoch  55  time  0.15556120872497559
Average time per epoch 0.09034234094619752 (for 500 epochs)
Epoch  56  time  0.15420293807983398
Average time per epoch 0.09065074682235717 (for 500 epochs)
Epoch  57  time  0.16852307319641113
Average time per epoch 0.09098779296875 (for 500 epochs)
Epoch  58  time  0.14613556861877441
Average time per epoch 0.09128006410598755 (for 500 epochs)
Epoch  59  time  0.1558682918548584
Average time per epoch 0.09159180068969727 (for 500 epochs)
Epoch  60  time  0.14417743682861328
Epoch  60  loss  2.8362409065109184 correct 49
Average time per epoch 0.0918801555633545 (for 500 epochs)
Epoch  61  time  0.1471264362335205
Average time per epoch 0.09217440843582153 (for 500 epochs)
Epoch  62  time  0.1504817008972168
Average time per epoch 0.09247537183761596 (for 500 epochs)
Epoch  63  time  0.1605687141418457
Average time per epoch 0.09279650926589966 (for 500 epochs)
Epoch  64  time  0.14600157737731934
Average time per epoch 0.0930885124206543 (for 500 epochs)
Epoch  65  time  0.149702787399292
Average time per epoch 0.09338791799545289 (for 500 epochs)
Epoch  66  time  0.1529548168182373
Average time per epoch 0.09369382762908936 (for 500 epochs)
Epoch  67  time  0.14685535430908203
Average time per epoch 0.09398753833770752 (for 500 epochs)
Epoch  68  time  0.14319539070129395
Average time per epoch 0.0942739291191101 (for 500 epochs)
Epoch  69  time  0.15234947204589844
Average time per epoch 0.0945786280632019 (for 500 epochs)
Epoch  70  time  0.16013813018798828
Epoch  70  loss  2.529280863745034 correct 49
Average time per epoch 0.09489890432357788 (for 500 epochs)
Epoch  71  time  0.15104913711547852
Average time per epoch 0.09520100259780884 (for 500 epochs)
Epoch  72  time  0.1470165252685547
Average time per epoch 0.09549503564834595 (for 500 epochs)
Epoch  73  time  0.15396904945373535
Average time per epoch 0.09580297374725341 (for 500 epochs)
Epoch  74  time  0.147843599319458
Average time per epoch 0.09609866094589234 (for 500 epochs)
Epoch  75  time  0.15378165245056152
Average time per epoch 0.09640622425079345 (for 500 epochs)
Epoch  76  time  0.15272045135498047
Average time per epoch 0.09671166515350342 (for 500 epochs)
Epoch  77  time  0.1674041748046875
Average time per epoch 0.0970464735031128 (for 500 epochs)
Epoch  78  time  0.15163469314575195
Average time per epoch 0.0973497428894043 (for 500 epochs)
Epoch  79  time  0.17795109748840332
Average time per epoch 0.09770564508438111 (for 500 epochs)
Epoch  80  time  0.15064430236816406
Epoch  80  loss  1.6846484268549167 correct 49
Average time per epoch 0.09800693368911743 (for 500 epochs)
Epoch  81  time  0.14580512046813965
Average time per epoch 0.09829854393005372 (for 500 epochs)
Epoch  82  time  0.15072846412658691
Average time per epoch 0.09860000085830689 (for 500 epochs)
Epoch  83  time  0.15910601615905762
Average time per epoch 0.098918212890625 (for 500 epochs)
Epoch  84  time  0.15097832679748535
Average time per epoch 0.09922016954421997 (for 500 epochs)
Epoch  85  time  0.14561223983764648
Average time per epoch 0.09951139402389526 (for 500 epochs)
Epoch  86  time  0.14589262008666992
Average time per epoch 0.0998031792640686 (for 500 epochs)
Epoch  87  time  0.14932632446289062
Average time per epoch 0.10010183191299439 (for 500 epochs)
Epoch  88  time  0.1488659381866455
Average time per epoch 0.10039956378936768 (for 500 epochs)
Epoch  89  time  0.15220117568969727
Average time per epoch 0.10070396614074707 (for 500 epochs)
Epoch  90  time  0.17026758193969727
Epoch  90  loss  1.8927880848024337 correct 50
Average time per epoch 0.10104450130462647 (for 500 epochs)
Epoch  91  time  0.14464950561523438
Average time per epoch 0.10133380031585694 (for 500 epochs)
Epoch  92  time  0.15172719955444336
Average time per epoch 0.10163725471496582 (for 500 epochs)
Epoch  93  time  0.1473252773284912
Average time per epoch 0.1019319052696228 (for 500 epochs)
Epoch  94  time  0.14980649948120117
Average time per epoch 0.1022315182685852 (for 500 epochs)
Epoch  95  time  0.14433979988098145
Average time per epoch 0.10252019786834717 (for 500 epochs)
Epoch  96  time  0.15640878677368164
Average time per epoch 0.10283301544189453 (for 500 epochs)
Epoch  97  time  0.15976309776306152
Average time per epoch 0.10315254163742066 (for 500 epochs)
Epoch  98  time  0.14977097511291504
Average time per epoch 0.10345208358764649 (for 500 epochs)
Epoch  99  time  0.14472246170043945
Average time per epoch 0.10374152851104736 (for 500 epochs)
Epoch  100  time  0.15527653694152832
Epoch  100  loss  1.6706788173620537 correct 49
Average time per epoch 0.10405208158493041 (for 500 epochs)
Epoch  101  time  0.1474609375
Average time per epoch 0.10434700345993042 (for 500 epochs)
Epoch  102  time  0.14938688278198242
Average time per epoch 0.10464577722549438 (for 500 epochs)
Epoch  103  time  0.16870999336242676
Average time per epoch 0.10498319721221924 (for 500 epochs)
Epoch  104  time  0.15195059776306152
Average time per epoch 0.10528709840774536 (for 500 epochs)
Epoch  105  time  0.14240074157714844
Average time per epoch 0.10557189989089966 (for 500 epochs)
Epoch  106  time  0.15053319931030273
Average time per epoch 0.10587296628952027 (for 500 epochs)
Epoch  107  time  0.1468040943145752
Average time per epoch 0.10616657447814941 (for 500 epochs)
Epoch  108  time  0.1518716812133789
Average time per epoch 0.10647031784057617 (for 500 epochs)
Epoch  109  time  0.14735722541809082
Average time per epoch 0.10676503229141235 (for 500 epochs)
Epoch  110  time  0.16331148147583008
Epoch  110  loss  0.8939541190961006 correct 49
Average time per epoch 0.10709165525436401 (for 500 epochs)
Epoch  111  time  0.14844274520874023
Average time per epoch 0.10738854074478149 (for 500 epochs)
Epoch  112  time  0.14800119400024414
Average time per epoch 0.10768454313278199 (for 500 epochs)
Epoch  113  time  0.14795207977294922
Average time per epoch 0.10798044729232788 (for 500 epochs)
Epoch  114  time  0.15464019775390625
Average time per epoch 0.1082897276878357 (for 500 epochs)
Epoch  115  time  0.1480264663696289
Average time per epoch 0.10858578062057495 (for 500 epochs)
Epoch  116  time  0.1615288257598877
Average time per epoch 0.10890883827209473 (for 500 epochs)
Epoch  117  time  0.1660919189453125
Average time per epoch 0.10924102210998535 (for 500 epochs)
Epoch  118  time  0.17324280738830566
Average time per epoch 0.10958750772476196 (for 500 epochs)
Epoch  119  time  0.14787840843200684
Average time per epoch 0.10988326454162597 (for 500 epochs)
Epoch  120  time  0.148529052734375
Epoch  120  loss  1.0988469947853086 correct 49
Average time per epoch 0.11018032264709472 (for 500 epochs)
Epoch  121  time  0.15515637397766113
Average time per epoch 0.11049063539505005 (for 500 epochs)
Epoch  122  time  0.15036368370056152
Average time per epoch 0.11079136276245118 (for 500 epochs)
Epoch  123  time  0.1620175838470459
Average time per epoch 0.11111539793014526 (for 500 epochs)
Epoch  124  time  0.15506815910339355
Average time per epoch 0.11142553424835205 (for 500 epochs)
Epoch  125  time  0.14097380638122559
Average time per epoch 0.1117074818611145 (for 500 epochs)
Epoch  126  time  0.14099502563476562
Average time per epoch 0.11198947191238404 (for 500 epochs)
Epoch  127  time  0.14469480514526367
Average time per epoch 0.11227886152267456 (for 500 epochs)
Epoch  128  time  0.14885401725769043
Average time per epoch 0.11257656955718995 (for 500 epochs)
Epoch  129  time  0.14963078498840332
Average time per epoch 0.11287583112716675 (for 500 epochs)
Epoch  130  time  0.16183042526245117
Epoch  130  loss  1.55156053026908 correct 49
Average time per epoch 0.11319949197769165 (for 500 epochs)
Epoch  131  time  0.15481042861938477
Average time per epoch 0.11350911283493043 (for 500 epochs)
Epoch  132  time  0.14791131019592285
Average time per epoch 0.11380493545532226 (for 500 epochs)
Epoch  133  time  0.14942336082458496
Average time per epoch 0.11410378217697144 (for 500 epochs)
Epoch  134  time  0.14665699005126953
Average time per epoch 0.11439709615707397 (for 500 epochs)
Epoch  135  time  0.15739989280700684
Average time per epoch 0.11471189594268799 (for 500 epochs)
Epoch  136  time  0.14186811447143555
Average time per epoch 0.11499563217163086 (for 500 epochs)
Epoch  137  time  0.18269729614257812
Average time per epoch 0.11536102676391602 (for 500 epochs)
Epoch  138  time  0.14535236358642578
Average time per epoch 0.11565173149108887 (for 500 epochs)
Epoch  139  time  0.14855432510375977
Average time per epoch 0.11594884014129639 (for 500 epochs)
Epoch  140  time  0.14810872077941895
Epoch  140  loss  0.9314653678327645 correct 49
Average time per epoch 0.11624505758285522 (for 500 epochs)
Epoch  141  time  0.14996886253356934
Average time per epoch 0.11654499530792237 (for 500 epochs)
Epoch  142  time  0.15509819984436035
Average time per epoch 0.11685519170761108 (for 500 epochs)
Epoch  143  time  0.16834521293640137
Average time per epoch 0.11719188213348389 (for 500 epochs)
Epoch  144  time  0.15181303024291992
Average time per epoch 0.11749550819396973 (for 500 epochs)
Epoch  145  time  0.15652060508728027
Average time per epoch 0.11780854940414429 (for 500 epochs)
Epoch  146  time  0.1696772575378418
Average time per epoch 0.11814790391921998 (for 500 epochs)
Epoch  147  time  0.15289020538330078
Average time per epoch 0.11845368432998657 (for 500 epochs)
Epoch  148  time  0.15238475799560547
Average time per epoch 0.11875845384597779 (for 500 epochs)
Epoch  149  time  0.1515042781829834
Average time per epoch 0.11906146240234375 (for 500 epochs)
Epoch  150  time  0.15805411338806152
Epoch  150  loss  0.3819402430688222 correct 49
Average time per epoch 0.11937757062911987 (for 500 epochs)
Epoch  151  time  0.1507568359375
Average time per epoch 0.11967908430099487 (for 500 epochs)
Epoch  152  time  0.15655040740966797
Average time per epoch 0.1199921851158142 (for 500 epochs)
Epoch  153  time  0.14817070960998535
Average time per epoch 0.12028852653503418 (for 500 epochs)
Epoch  154  time  0.15138936042785645
Average time per epoch 0.12059130525588989 (for 500 epochs)
Epoch  155  time  0.1597881317138672
Average time per epoch 0.12091088151931763 (for 500 epochs)
Epoch  156  time  0.16574645042419434
Average time per epoch 0.12124237442016601 (for 500 epochs)
Epoch  157  time  0.15164470672607422
Average time per epoch 0.12154566383361816 (for 500 epochs)
Epoch  158  time  0.14600205421447754
Average time per epoch 0.12183766794204712 (for 500 epochs)
Epoch  159  time  0.13950228691101074
Average time per epoch 0.12211667251586913 (for 500 epochs)
Epoch  160  time  0.15146136283874512
Epoch  160  loss  0.6390843186250478 correct 49
Average time per epoch 0.12241959524154664 (for 500 epochs)
Epoch  161  time  0.14516019821166992
Average time per epoch 0.12270991563796997 (for 500 epochs)
Epoch  162  time  0.15694952011108398
Average time per epoch 0.12302381467819214 (for 500 epochs)
Epoch  163  time  0.16209077835083008
Average time per epoch 0.1233479962348938 (for 500 epochs)
Epoch  164  time  0.15783357620239258
Average time per epoch 0.12366366338729859 (for 500 epochs)
Epoch  165  time  0.14937782287597656
Average time per epoch 0.12396241903305054 (for 500 epochs)
Epoch  166  time  0.15437531471252441
Average time per epoch 0.12427116966247559 (for 500 epochs)
Epoch  167  time  0.1535017490386963
Average time per epoch 0.12457817316055297 (for 500 epochs)
Epoch  168  time  0.15728330612182617
Average time per epoch 0.12489273977279663 (for 500 epochs)
Epoch  169  time  0.15106749534606934
Average time per epoch 0.12519487476348876 (for 500 epochs)
Epoch  170  time  0.17015314102172852
Epoch  170  loss  0.5502554019058042 correct 49
Average time per epoch 0.12553518104553224 (for 500 epochs)
Epoch  171  time  0.15163493156433105
Average time per epoch 0.1258384509086609 (for 500 epochs)
Epoch  172  time  0.15540170669555664
Average time per epoch 0.126149254322052 (for 500 epochs)
Epoch  173  time  0.15243887901306152
Average time per epoch 0.12645413208007814 (for 500 epochs)
Epoch  174  time  0.1536116600036621
Average time per epoch 0.12676135540008546 (for 500 epochs)
Epoch  175  time  0.1494462490081787
Average time per epoch 0.1270602478981018 (for 500 epochs)
Epoch  176  time  0.16339969635009766
Average time per epoch 0.127387047290802 (for 500 epochs)
Epoch  177  time  0.14238953590393066
Average time per epoch 0.12767182636260987 (for 500 epochs)
Epoch  178  time  0.1459202766418457
Average time per epoch 0.12796366691589356 (for 500 epochs)
Epoch  179  time  0.15365886688232422
Average time per epoch 0.1282709846496582 (for 500 epochs)
Epoch  180  time  0.1468949317932129
Epoch  180  loss  0.5243102200993456 correct 49
Average time per epoch 0.12856477451324463 (for 500 epochs)
Epoch  181  time  0.15329742431640625
Average time per epoch 0.12887136936187743 (for 500 epochs)
Epoch  182  time  0.15239477157592773
Average time per epoch 0.1291761589050293 (for 500 epochs)
Epoch  183  time  0.15430164337158203
Average time per epoch 0.12948476219177246 (for 500 epochs)
Epoch  184  time  0.14726543426513672
Average time per epoch 0.12977929306030273 (for 500 epochs)
Epoch  185  time  0.14898943901062012
Average time per epoch 0.13007727193832397 (for 500 epochs)
Epoch  186  time  0.14879107475280762
Average time per epoch 0.1303748540878296 (for 500 epochs)
Epoch  187  time  0.1499772071838379
Average time per epoch 0.13067480850219726 (for 500 epochs)
Epoch  188  time  0.14567875862121582
Average time per epoch 0.1309661660194397 (for 500 epochs)
Epoch  189  time  0.14642786979675293
Average time per epoch 0.1312590217590332 (for 500 epochs)
Epoch  190  time  0.16008806228637695
Epoch  190  loss  0.7093006173358136 correct 49
Average time per epoch 0.13157919788360595 (for 500 epochs)
Epoch  191  time  0.14438509941101074
Average time per epoch 0.13186796808242798 (for 500 epochs)
Epoch  192  time  0.1567995548248291
Average time per epoch 0.13218156719207763 (for 500 epochs)
Epoch  193  time  0.14944148063659668
Average time per epoch 0.13248045015335083 (for 500 epochs)
Epoch  194  time  0.15792369842529297
Average time per epoch 0.13279629755020142 (for 500 epochs)
Epoch  195  time  0.14689946174621582
Average time per epoch 0.13309009647369385 (for 500 epochs)
Epoch  196  time  0.17054033279418945
Average time per epoch 0.13343117713928224 (for 500 epochs)
Epoch  197  time  0.1480238437652588
Average time per epoch 0.13372722482681274 (for 500 epochs)
Epoch  198  time  0.15263748168945312
Average time per epoch 0.13403249979019166 (for 500 epochs)
Epoch  199  time  0.14209723472595215
Average time per epoch 0.13431669425964354 (for 500 epochs)
Epoch  200  time  0.14926433563232422
Epoch  200  loss  0.6304643608647728 correct 49
Average time per epoch 0.1346152229309082 (for 500 epochs)
Epoch  201  time  0.16159439086914062
Average time per epoch 0.13493841171264648 (for 500 epochs)
Epoch  202  time  0.16466212272644043
Average time per epoch 0.13526773595809938 (for 500 epochs)
Epoch  203  time  0.15643978118896484
Average time per epoch 0.1355806155204773 (for 500 epochs)
Epoch  204  time  0.14544200897216797
Average time per epoch 0.13587149953842162 (for 500 epochs)
Epoch  205  time  0.14507174491882324
Average time per epoch 0.13616164302825928 (for 500 epochs)
Epoch  206  time  0.1594228744506836
Average time per epoch 0.13648048877716065 (for 500 epochs)
Epoch  207  time  0.14433789253234863
Average time per epoch 0.13676916456222535 (for 500 epochs)
Epoch  208  time  0.15371322631835938
Average time per epoch 0.13707659101486205 (for 500 epochs)
Epoch  209  time  0.14971423149108887
Average time per epoch 0.13737601947784422 (for 500 epochs)
Epoch  210  time  0.1644442081451416
Epoch  210  loss  0.6286110998596873 correct 50
Average time per epoch 0.13770490789413453 (for 500 epochs)
Epoch  211  time  0.1488478183746338
Average time per epoch 0.13800260353088378 (for 500 epochs)
Epoch  212  time  0.17021560668945312
Average time per epoch 0.13834303474426268 (for 500 epochs)
Epoch  213  time  0.14507317543029785
Average time per epoch 0.1386331810951233 (for 500 epochs)
Epoch  214  time  0.15526080131530762
Average time per epoch 0.1389437026977539 (for 500 epochs)
Epoch  215  time  0.14763593673706055
Average time per epoch 0.13923897457122802 (for 500 epochs)
Epoch  216  time  0.16160178184509277
Average time per epoch 0.13956217813491822 (for 500 epochs)
Epoch  217  time  0.14404559135437012
Average time per epoch 0.13985026931762695 (for 500 epochs)
Epoch  218  time  0.1496880054473877
Average time per epoch 0.14014964532852173 (for 500 epochs)
Epoch  219  time  0.13973093032836914
Average time per epoch 0.14042910718917848 (for 500 epochs)
Epoch  220  time  0.15535521507263184
Epoch  220  loss  0.5204436917833745 correct 49
Average time per epoch 0.14073981761932372 (for 500 epochs)
Epoch  221  time  0.1477663516998291
Average time per epoch 0.14103535032272338 (for 500 epochs)
Epoch  222  time  0.15432429313659668
Average time per epoch 0.14134399890899657 (for 500 epochs)
Epoch  223  time  0.16064715385437012
Average time per epoch 0.14166529321670532 (for 500 epochs)
Epoch  224  time  0.14865994453430176
Average time per epoch 0.14196261310577393 (for 500 epochs)
Epoch  225  time  0.16781949996948242
Average time per epoch 0.1422982521057129 (for 500 epochs)
Epoch  226  time  0.17266607284545898
Average time per epoch 0.1426435842514038 (for 500 epochs)
Epoch  227  time  0.15091395378112793
Average time per epoch 0.14294541215896606 (for 500 epochs)
Epoch  228  time  0.14910054206848145
Average time per epoch 0.14324361324310303 (for 500 epochs)
Epoch  229  time  0.15938735008239746
Average time per epoch 0.1435623879432678 (for 500 epochs)
Epoch  230  time  0.14803719520568848
Epoch  230  loss  1.0228578697475392 correct 49
Average time per epoch 0.1438584623336792 (for 500 epochs)
Epoch  231  time  0.14456653594970703
Average time per epoch 0.14414759540557862 (for 500 epochs)
Epoch  232  time  0.1553950309753418
Average time per epoch 0.1444583854675293 (for 500 epochs)
Epoch  233  time  0.1439652442932129
Average time per epoch 0.14474631595611573 (for 500 epochs)
Epoch  234  time  0.14678072929382324
Average time per epoch 0.14503987741470337 (for 500 epochs)
Epoch  235  time  0.15240263938903809
Average time per epoch 0.14534468269348144 (for 500 epochs)
Epoch  236  time  0.15590405464172363
Average time per epoch 0.14565649080276488 (for 500 epochs)
Epoch  237  time  0.16256284713745117
Average time per epoch 0.1459816164970398 (for 500 epochs)
Epoch  238  time  0.1437373161315918
Average time per epoch 0.14626909112930297 (for 500 epochs)
Epoch  239  time  0.14688634872436523
Average time per epoch 0.1465628638267517 (for 500 epochs)
Epoch  240  time  0.15308690071105957
Epoch  240  loss  1.0938197863119565 correct 50
Average time per epoch 0.14686903762817383 (for 500 epochs)
Epoch  241  time  0.15016770362854004
Average time per epoch 0.1471693730354309 (for 500 epochs)
Epoch  242  time  0.14565372467041016
Average time per epoch 0.14746068048477173 (for 500 epochs)
Epoch  243  time  0.1631031036376953
Average time per epoch 0.14778688669204712 (for 500 epochs)
Epoch  244  time  0.15009021759033203
Average time per epoch 0.14808706712722777 (for 500 epochs)
Epoch  245  time  0.15769267082214355
Average time per epoch 0.14840245246887207 (for 500 epochs)
Epoch  246  time  0.1430654525756836
Average time per epoch 0.14868858337402344 (for 500 epochs)
Epoch  247  time  0.15155720710754395
Average time per epoch 0.14899169778823854 (for 500 epochs)
Epoch  248  time  0.14553546905517578
Average time per epoch 0.14928276872634888 (for 500 epochs)
Epoch  249  time  0.16434526443481445
Average time per epoch 0.14961145925521852 (for 500 epochs)
Epoch  250  time  0.15508699417114258
Epoch  250  loss  1.237183195830972 correct 50
Average time per epoch 0.14992163324356078 (for 500 epochs)
Epoch  251  time  0.15353178977966309
Average time per epoch 0.15022869682312012 (for 500 epochs)
Epoch  252  time  0.14394927024841309
Average time per epoch 0.15051659536361695 (for 500 epochs)
Epoch  253  time  0.14864778518676758
Average time per epoch 0.15081389093399047 (for 500 epochs)
Epoch  254  time  0.15753722190856934
Average time per epoch 0.15112896537780762 (for 500 epochs)
Epoch  255  time  0.15273809432983398
Average time per epoch 0.1514344415664673 (for 500 epochs)
Epoch  256  time  0.1614093780517578
Average time per epoch 0.1517572603225708 (for 500 epochs)
Epoch  257  time  0.14656782150268555
Average time per epoch 0.15205039596557618 (for 500 epochs)
Epoch  258  time  0.15013670921325684
Average time per epoch 0.15235066938400268 (for 500 epochs)
Epoch  259  time  0.14782261848449707
Average time per epoch 0.15264631462097167 (for 500 epochs)
Epoch  260  time  0.15527105331420898
Epoch  260  loss  0.2814536111111162 correct 50
Average time per epoch 0.1529568567276001 (for 500 epochs)
Epoch  261  time  0.14573383331298828
Average time per epoch 0.1532483243942261 (for 500 epochs)
Epoch  262  time  0.1488943099975586
Average time per epoch 0.1535461130142212 (for 500 epochs)
Epoch  263  time  0.15685725212097168
Average time per epoch 0.15385982751846314 (for 500 epochs)
Epoch  264  time  0.1495835781097412
Average time per epoch 0.1541589946746826 (for 500 epochs)
Epoch  265  time  0.14820432662963867
Average time per epoch 0.1544554033279419 (for 500 epochs)
Epoch  266  time  0.15598154067993164
Average time per epoch 0.15476736640930175 (for 500 epochs)
Epoch  267  time  0.15184307098388672
Average time per epoch 0.15507105255126954 (for 500 epochs)
Epoch  268  time  0.1460270881652832
Average time per epoch 0.1553631067276001 (for 500 epochs)
Epoch  269  time  0.1411750316619873
Average time per epoch 0.15564545679092406 (for 500 epochs)
Epoch  270  time  0.16283369064331055
Epoch  270  loss  0.22619247887593621 correct 50
Average time per epoch 0.1559711241722107 (for 500 epochs)
Epoch  271  time  0.14842939376831055
Average time per epoch 0.1562679829597473 (for 500 epochs)
Epoch  272  time  0.1473388671875
Average time per epoch 0.1565626606941223 (for 500 epochs)
Epoch  273  time  0.14612889289855957
Average time per epoch 0.15685491847991945 (for 500 epochs)
Epoch  274  time  0.14733481407165527
Average time per epoch 0.15714958810806273 (for 500 epochs)
Epoch  275  time  0.1479787826538086
Average time per epoch 0.15744554567337035 (for 500 epochs)
Epoch  276  time  0.16653966903686523
Average time per epoch 0.1577786250114441 (for 500 epochs)
Epoch  277  time  0.1564629077911377
Average time per epoch 0.15809155082702636 (for 500 epochs)
Epoch  278  time  0.15422391891479492
Average time per epoch 0.15839999866485596 (for 500 epochs)
Epoch  279  time  0.14772820472717285
Average time per epoch 0.1586954550743103 (for 500 epochs)
Epoch  280  time  0.19755053520202637
Epoch  280  loss  1.4653347488280615 correct 49
Average time per epoch 0.15909055614471435 (for 500 epochs)
Epoch  281  time  0.14481878280639648
Average time per epoch 0.15938019371032716 (for 500 epochs)
Epoch  282  time  0.1488032341003418
Average time per epoch 0.15967780017852784 (for 500 epochs)
Epoch  283  time  0.16297554969787598
Average time per epoch 0.16000375127792357 (for 500 epochs)
Epoch  284  time  0.15965962409973145
Average time per epoch 0.16032307052612305 (for 500 epochs)
Epoch  285  time  0.3763751983642578
Average time per epoch 0.16107582092285155 (for 500 epochs)
Epoch  286  time  2.5471184253692627
Average time per epoch 0.1661700577735901 (for 500 epochs)
Epoch  287  time  1.1711921691894531
Average time per epoch 0.168512442111969 (for 500 epochs)
Epoch  288  time  1.5549416542053223
Average time per epoch 0.17162232542037964 (for 500 epochs)
Epoch  289  time  0.6086776256561279
Average time per epoch 0.1728396806716919 (for 500 epochs)
Epoch  290  time  1.8198580741882324
Epoch  290  loss  0.3904365227621004 correct 49
Average time per epoch 0.17647939682006836 (for 500 epochs)
Epoch  291  time  0.49092674255371094
Average time per epoch 0.1774612503051758 (for 500 epochs)
Epoch  292  time  1.385725975036621
Average time per epoch 0.180232702255249 (for 500 epochs)
Epoch  293  time  1.1714491844177246
Average time per epoch 0.18257560062408448 (for 500 epochs)
Epoch  294  time  0.5755317211151123
Average time per epoch 0.1837266640663147 (for 500 epochs)
Epoch  295  time  0.154158353805542
Average time per epoch 0.18403498077392577 (for 500 epochs)
Epoch  296  time  0.15324664115905762
Average time per epoch 0.1843414740562439 (for 500 epochs)
Epoch  297  time  0.15985703468322754
Average time per epoch 0.18466118812561036 (for 500 epochs)
Epoch  298  time  0.14902400970458984
Average time per epoch 0.18495923614501952 (for 500 epochs)
Epoch  299  time  0.13950204849243164
Average time per epoch 0.1852382402420044 (for 500 epochs)
Epoch  300  time  0.15199041366577148
Epoch  300  loss  0.062368118555333585 correct 49
Average time per epoch 0.18554222106933593 (for 500 epochs)
Epoch  301  time  0.14603209495544434
Average time per epoch 0.18583428525924683 (for 500 epochs)
Epoch  302  time  0.1523451805114746
Average time per epoch 0.18613897562026976 (for 500 epochs)
Epoch  303  time  0.147413969039917
Average time per epoch 0.1864338035583496 (for 500 epochs)
Epoch  304  time  0.16422486305236816
Average time per epoch 0.18676225328445434 (for 500 epochs)
Epoch  305  time  0.14905738830566406
Average time per epoch 0.18706036806106568 (for 500 epochs)
Epoch  306  time  0.14518356323242188
Average time per epoch 0.1873507351875305 (for 500 epochs)
Epoch  307  time  0.14653539657592773
Average time per epoch 0.18764380598068237 (for 500 epochs)
Epoch  308  time  0.15267300605773926
Average time per epoch 0.18794915199279785 (for 500 epochs)
Epoch  309  time  0.1448986530303955
Average time per epoch 0.18823894929885865 (for 500 epochs)
Epoch  310  time  0.14786291122436523
Epoch  310  loss  0.9453952325718947 correct 49
Average time per epoch 0.18853467512130737 (for 500 epochs)
Epoch  311  time  0.15962433815002441
Average time per epoch 0.18885392379760743 (for 500 epochs)
Epoch  312  time  0.1473245620727539
Average time per epoch 0.18914857292175294 (for 500 epochs)
Epoch  313  time  0.14258027076721191
Average time per epoch 0.18943373346328735 (for 500 epochs)
Epoch  314  time  0.14667224884033203
Average time per epoch 0.18972707796096802 (for 500 epochs)
Epoch  315  time  0.14424943923950195
Average time per epoch 0.19001557683944703 (for 500 epochs)
Epoch  316  time  0.1559591293334961
Average time per epoch 0.190327495098114 (for 500 epochs)
Epoch  317  time  0.1623239517211914
Average time per epoch 0.1906521430015564 (for 500 epochs)
Epoch  318  time  0.1491391658782959
Average time per epoch 0.190950421333313 (for 500 epochs)
Epoch  319  time  0.1444382667541504
Average time per epoch 0.19123929786682128 (for 500 epochs)
Epoch  320  time  0.15222907066345215
Epoch  320  loss  0.2429817732975276 correct 50
Average time per epoch 0.19154375600814819 (for 500 epochs)
Epoch  321  time  0.14489269256591797
Average time per epoch 0.19183354139328002 (for 500 epochs)
Epoch  322  time  0.16868352890014648
Average time per epoch 0.1921709084510803 (for 500 epochs)
Epoch  323  time  0.14400434494018555
Average time per epoch 0.1924589171409607 (for 500 epochs)
Epoch  324  time  0.1645948886871338
Average time per epoch 0.19278810691833495 (for 500 epochs)
Epoch  325  time  0.14152789115905762
Average time per epoch 0.19307116270065308 (for 500 epochs)
Epoch  326  time  0.15856480598449707
Average time per epoch 0.19338829231262206 (for 500 epochs)
Epoch  327  time  0.14464640617370605
Average time per epoch 0.1936775851249695 (for 500 epochs)
Epoch  328  time  0.15570545196533203
Average time per epoch 0.19398899602890016 (for 500 epochs)
Epoch  329  time  0.1492936611175537
Average time per epoch 0.19428758335113525 (for 500 epochs)
Epoch  330  time  0.15160417556762695
Epoch  330  loss  0.22371239705042642 correct 50
Average time per epoch 0.1945907917022705 (for 500 epochs)
Epoch  331  time  0.16493773460388184
Average time per epoch 0.19492066717147827 (for 500 epochs)
Epoch  332  time  0.20804977416992188
Average time per epoch 0.19533676671981812 (for 500 epochs)
Epoch  333  time  0.14484739303588867
Average time per epoch 0.1956264615058899 (for 500 epochs)
Epoch  334  time  0.1557328701019287
Average time per epoch 0.19593792724609374 (for 500 epochs)
Epoch  335  time  0.14227056503295898
Average time per epoch 0.19622246837615967 (for 500 epochs)
Epoch  336  time  0.1551520824432373
Average time per epoch 0.19653277254104615 (for 500 epochs)
Epoch  337  time  0.15550756454467773
Average time per epoch 0.1968437876701355 (for 500 epochs)
Epoch  338  time  0.14699244499206543
Average time per epoch 0.19713777256011963 (for 500 epochs)
Epoch  339  time  0.13977742195129395
Average time per epoch 0.1974173274040222 (for 500 epochs)
Epoch  340  time  0.16166234016418457
Epoch  340  loss  0.16823003434278605 correct 50
Average time per epoch 0.1977406520843506 (for 500 epochs)
Epoch  341  time  0.16785073280334473
Average time per epoch 0.19807635354995728 (for 500 epochs)
Epoch  342  time  0.17479205131530762
Average time per epoch 0.1984259376525879 (for 500 epochs)
Epoch  343  time  0.1758260726928711
Average time per epoch 0.19877758979797364 (for 500 epochs)
Epoch  344  time  0.15052509307861328
Average time per epoch 0.19907863998413086 (for 500 epochs)
Epoch  345  time  0.16007494926452637
Average time per epoch 0.1993987898826599 (for 500 epochs)
Epoch  346  time  0.15457510948181152
Average time per epoch 0.19970794010162354 (for 500 epochs)
Epoch  347  time  0.1592090129852295
Average time per epoch 0.200026358127594 (for 500 epochs)
Epoch  348  time  0.15391230583190918
Average time per epoch 0.20033418273925782 (for 500 epochs)
Epoch  349  time  0.25316667556762695
Average time per epoch 0.20084051609039308 (for 500 epochs)
Epoch  350  time  0.4938948154449463
Epoch  350  loss  0.7933427179574764 correct 49
Average time per epoch 0.20182830572128296 (for 500 epochs)
Epoch  351  time  1.662308931350708
Average time per epoch 0.20515292358398438 (for 500 epochs)
Epoch  352  time  2.0084149837493896
Average time per epoch 0.20916975355148315 (for 500 epochs)
Epoch  353  time  2.1214799880981445
Average time per epoch 0.21341271352767943 (for 500 epochs)
Epoch  354  time  1.2846927642822266
Average time per epoch 0.2159820990562439 (for 500 epochs)
Epoch  355  time  1.7981576919555664
Average time per epoch 0.21957841444015502 (for 500 epochs)
Epoch  356  time  0.4670860767364502
Average time per epoch 0.22051258659362794 (for 500 epochs)
Epoch  357  time  1.8515739440917969
Average time per epoch 0.22421573448181153 (for 500 epochs)
Epoch  358  time  0.1479949951171875
Average time per epoch 0.2245117244720459 (for 500 epochs)
Epoch  359  time  0.15205860137939453
Average time per epoch 0.2248158416748047 (for 500 epochs)
Epoch  360  time  0.14363312721252441
Epoch  360  loss  0.07567906089341991 correct 49
Average time per epoch 0.22510310792922975 (for 500 epochs)
Epoch  361  time  0.16505813598632812
Average time per epoch 0.2254332242012024 (for 500 epochs)
Epoch  362  time  0.14331483840942383
Average time per epoch 0.22571985387802124 (for 500 epochs)
Epoch  363  time  0.14691781997680664
Average time per epoch 0.22601368951797485 (for 500 epochs)
Epoch  364  time  0.15162277221679688
Average time per epoch 0.22631693506240844 (for 500 epochs)
Epoch  365  time  0.1463000774383545
Average time per epoch 0.22660953521728516 (for 500 epochs)
Epoch  366  time  0.15059614181518555
Average time per epoch 0.22691072750091554 (for 500 epochs)
Epoch  367  time  0.14937663078308105
Average time per epoch 0.22720948076248168 (for 500 epochs)
Epoch  368  time  0.16353130340576172
Average time per epoch 0.22753654336929321 (for 500 epochs)
Epoch  369  time  0.15098905563354492
Average time per epoch 0.2278385214805603 (for 500 epochs)
Epoch  370  time  0.16150426864624023
Epoch  370  loss  0.783583403949952 correct 49
Average time per epoch 0.22816153001785278 (for 500 epochs)
Epoch  371  time  0.15312933921813965
Average time per epoch 0.22846778869628906 (for 500 epochs)
Epoch  372  time  0.16878509521484375
Average time per epoch 0.22880535888671874 (for 500 epochs)
Epoch  373  time  0.14602303504943848
Average time per epoch 0.22909740495681763 (for 500 epochs)
Epoch  374  time  0.14966201782226562
Average time per epoch 0.22939672899246216 (for 500 epochs)
Epoch  375  time  0.16442227363586426
Average time per epoch 0.22972557353973388 (for 500 epochs)
Epoch  376  time  0.15434932708740234
Average time per epoch 0.23003427219390868 (for 500 epochs)
Epoch  377  time  0.14358973503112793
Average time per epoch 0.23032145166397094 (for 500 epochs)
Epoch  378  time  0.14786314964294434
Average time per epoch 0.23061717796325684 (for 500 epochs)
Epoch  379  time  0.14920473098754883
Average time per epoch 0.23091558742523194 (for 500 epochs)
Epoch  380  time  0.14454030990600586
Epoch  380  loss  1.276410478531942 correct 49
Average time per epoch 0.23120466804504394 (for 500 epochs)
Epoch  381  time  0.15935063362121582
Average time per epoch 0.23152336931228637 (for 500 epochs)
Epoch  382  time  0.1542503833770752
Average time per epoch 0.23183187007904052 (for 500 epochs)
Epoch  383  time  0.1479930877685547
Average time per epoch 0.23212785625457763 (for 500 epochs)
Epoch  384  time  0.15366053581237793
Average time per epoch 0.2324351773262024 (for 500 epochs)
Epoch  385  time  0.1517195701599121
Average time per epoch 0.23273861646652222 (for 500 epochs)
Epoch  386  time  0.14998173713684082
Average time per epoch 0.2330385799407959 (for 500 epochs)
Epoch  387  time  0.15070271492004395
Average time per epoch 0.233339985370636 (for 500 epochs)
Epoch  388  time  0.1614820957183838
Average time per epoch 0.23366294956207276 (for 500 epochs)
Epoch  389  time  0.16385579109191895
Average time per epoch 0.2339906611442566 (for 500 epochs)
Epoch  390  time  0.14347314834594727
Epoch  390  loss  0.11247913894555221 correct 49
Average time per epoch 0.2342776074409485 (for 500 epochs)
Epoch  391  time  0.1524336338043213
Average time per epoch 0.23458247470855714 (for 500 epochs)
Epoch  392  time  0.14493775367736816
Average time per epoch 0.23487235021591185 (for 500 epochs)
Epoch  393  time  0.15619230270385742
Average time per epoch 0.23518473482131957 (for 500 epochs)
Epoch  394  time  0.1461505889892578
Average time per epoch 0.2354770359992981 (for 500 epochs)
Epoch  395  time  0.1665666103363037
Average time per epoch 0.2358101692199707 (for 500 epochs)
Epoch  396  time  0.1485767364501953
Average time per epoch 0.2361073226928711 (for 500 epochs)
Epoch  397  time  0.1625504493713379
Average time per epoch 0.23643242359161376 (for 500 epochs)
Epoch  398  time  0.1468489170074463
Average time per epoch 0.23672612142562866 (for 500 epochs)
Epoch  399  time  0.15824508666992188
Average time per epoch 0.2370426115989685 (for 500 epochs)
Epoch  400  time  0.14482998847961426
Epoch  400  loss  0.8720374343584159 correct 49
Average time per epoch 0.23733227157592773 (for 500 epochs)
Epoch  401  time  0.16657161712646484
Average time per epoch 0.23766541481018066 (for 500 epochs)
Epoch  402  time  0.14520502090454102
Average time per epoch 0.23795582485198974 (for 500 epochs)
Epoch  403  time  0.1517350673675537
Average time per epoch 0.23825929498672485 (for 500 epochs)
Epoch  404  time  0.15097570419311523
Average time per epoch 0.2385612463951111 (for 500 epochs)
Epoch  405  time  0.15011286735534668
Average time per epoch 0.23886147212982178 (for 500 epochs)
Epoch  406  time  0.14314794540405273
Average time per epoch 0.23914776802062987 (for 500 epochs)
Epoch  407  time  0.17950844764709473
Average time per epoch 0.23950678491592406 (for 500 epochs)
Epoch  408  time  0.17595767974853516
Average time per epoch 0.23985870027542114 (for 500 epochs)
Epoch  409  time  0.15722107887268066
Average time per epoch 0.2401731424331665 (for 500 epochs)
Epoch  410  time  0.1428534984588623
Epoch  410  loss  0.012971983292320698 correct 49
Average time per epoch 0.24045884943008422 (for 500 epochs)
Epoch  411  time  0.15131807327270508
Average time per epoch 0.24076148557662963 (for 500 epochs)
Epoch  412  time  0.14154958724975586
Average time per epoch 0.24104458475112914 (for 500 epochs)
Epoch  413  time  0.15820693969726562
Average time per epoch 0.24136099863052368 (for 500 epochs)
Epoch  414  time  0.14201021194458008
Average time per epoch 0.24164501905441285 (for 500 epochs)
Epoch  415  time  0.16512799263000488
Average time per epoch 0.24197527503967284 (for 500 epochs)
Epoch  416  time  0.14642882347106934
Average time per epoch 0.24226813268661498 (for 500 epochs)
Epoch  417  time  0.1579577922821045
Average time per epoch 0.2425840482711792 (for 500 epochs)
Epoch  418  time  0.14598560333251953
Average time per epoch 0.24287601947784423 (for 500 epochs)
Epoch  419  time  0.15375638008117676
Average time per epoch 0.2431835322380066 (for 500 epochs)
Epoch  420  time  0.17696309089660645
Epoch  420  loss  1.0214741363648223 correct 49
Average time per epoch 0.24353745841979982 (for 500 epochs)
Epoch  421  time  0.16289687156677246
Average time per epoch 0.24386325216293334 (for 500 epochs)
Epoch  422  time  0.14331531524658203
Average time per epoch 0.2441498827934265 (for 500 epochs)
Epoch  423  time  0.14863061904907227
Average time per epoch 0.24444714403152465 (for 500 epochs)
Epoch  424  time  0.14760327339172363
Average time per epoch 0.2447423505783081 (for 500 epochs)
Epoch  425  time  0.15015506744384766
Average time per epoch 0.2450426607131958 (for 500 epochs)
Epoch  426  time  0.14389395713806152
Average time per epoch 0.24533044862747191 (for 500 epochs)
Epoch  427  time  0.1554262638092041
Average time per epoch 0.24564130115509034 (for 500 epochs)
Epoch  428  time  0.1555953025817871
Average time per epoch 0.2459524917602539 (for 500 epochs)
Epoch  429  time  0.15444159507751465
Average time per epoch 0.24626137495040892 (for 500 epochs)
Epoch  430  time  0.14524245262145996
Epoch  430  loss  0.205368631163538 correct 49
Average time per epoch 0.24655185985565187 (for 500 epochs)
Epoch  431  time  0.14573168754577637
Average time per epoch 0.2468433232307434 (for 500 epochs)
Epoch  432  time  0.1460733413696289
Average time per epoch 0.24713546991348267 (for 500 epochs)
Epoch  433  time  0.1469101905822754
Average time per epoch 0.24742929029464722 (for 500 epochs)
Epoch  434  time  0.15148186683654785
Average time per epoch 0.2477322540283203 (for 500 epochs)
Epoch  435  time  0.16441750526428223
Average time per epoch 0.24806108903884888 (for 500 epochs)
Epoch  436  time  0.14417362213134766
Average time per epoch 0.24834943628311157 (for 500 epochs)
Epoch  437  time  0.15686869621276855
Average time per epoch 0.2486631736755371 (for 500 epochs)
Epoch  438  time  0.1507880687713623
Average time per epoch 0.24896474981307984 (for 500 epochs)
Epoch  439  time  0.1485307216644287
Average time per epoch 0.2492618112564087 (for 500 epochs)
Epoch  440  time  0.14887785911560059
Epoch  440  loss  1.2835464854566665 correct 49
Average time per epoch 0.24955956697463988 (for 500 epochs)
Epoch  441  time  0.16222810745239258
Average time per epoch 0.24988402318954467 (for 500 epochs)
Epoch  442  time  0.14097952842712402
Average time per epoch 0.2501659822463989 (for 500 epochs)
Epoch  443  time  0.14814209938049316
Average time per epoch 0.2504622664451599 (for 500 epochs)
Epoch  444  time  0.1400589942932129
Average time per epoch 0.25074238443374636 (for 500 epochs)
Epoch  445  time  0.14412403106689453
Average time per epoch 0.25103063249588015 (for 500 epochs)
Epoch  446  time  0.1403038501739502
Average time per epoch 0.251311240196228 (for 500 epochs)
Epoch  447  time  0.14834856986999512
Average time per epoch 0.25160793733596803 (for 500 epochs)
Epoch  448  time  0.15658354759216309
Average time per epoch 0.25192110443115234 (for 500 epochs)
Epoch  449  time  0.15353155136108398
Average time per epoch 0.2522281675338745 (for 500 epochs)
Epoch  450  time  0.14504432678222656
Epoch  450  loss  0.23247374172774743 correct 50
Average time per epoch 0.25251825618743895 (for 500 epochs)
Epoch  451  time  0.1532750129699707
Average time per epoch 0.2528248062133789 (for 500 epochs)
Epoch  452  time  0.1427288055419922
Average time per epoch 0.2531102638244629 (for 500 epochs)
Epoch  453  time  0.14647197723388672
Average time per epoch 0.25340320777893066 (for 500 epochs)
Epoch  454  time  0.1513066291809082
Average time per epoch 0.2537058210372925 (for 500 epochs)
Epoch  455  time  0.16120195388793945
Average time per epoch 0.2540282249450684 (for 500 epochs)
Epoch  456  time  0.16271495819091797
Average time per epoch 0.2543536548614502 (for 500 epochs)
Epoch  457  time  0.1441655158996582
Average time per epoch 0.2546419858932495 (for 500 epochs)
Epoch  458  time  0.139906644821167
Average time per epoch 0.2549217991828919 (for 500 epochs)
Epoch  459  time  0.1508464813232422
Average time per epoch 0.25522349214553836 (for 500 epochs)
Epoch  460  time  0.14422941207885742
Epoch  460  loss  0.0798251739761389 correct 49
Average time per epoch 0.25551195096969603 (for 500 epochs)
Epoch  461  time  0.16580796241760254
Average time per epoch 0.25584356689453125 (for 500 epochs)
Epoch  462  time  0.15888118743896484
Average time per epoch 0.25616132926940915 (for 500 epochs)
Epoch  463  time  0.14726710319519043
Average time per epoch 0.2564558634757996 (for 500 epochs)
Epoch  464  time  0.13950800895690918
Average time per epoch 0.2567348794937134 (for 500 epochs)
Epoch  465  time  0.15562748908996582
Average time per epoch 0.2570461344718933 (for 500 epochs)
Epoch  466  time  0.14255690574645996
Average time per epoch 0.2573312482833862 (for 500 epochs)
Epoch  467  time  0.14742279052734375
Average time per epoch 0.2576260938644409 (for 500 epochs)
Epoch  468  time  0.15002846717834473
Average time per epoch 0.2579261507987976 (for 500 epochs)
Epoch  469  time  0.1635115146636963
Average time per epoch 0.258253173828125 (for 500 epochs)
Epoch  470  time  0.14544296264648438
Epoch  470  loss  0.19342560263294556 correct 49
Average time per epoch 0.258544059753418 (for 500 epochs)
Epoch  471  time  0.1474323272705078
Average time per epoch 0.25883892440795897 (for 500 epochs)
Epoch  472  time  0.14984512329101562
Average time per epoch 0.259138614654541 (for 500 epochs)
Epoch  473  time  0.14331603050231934
Average time per epoch 0.2594252467155457 (for 500 epochs)
Epoch  474  time  0.14420413970947266
Average time per epoch 0.2597136549949646 (for 500 epochs)
Epoch  475  time  0.16827392578125
Average time per epoch 0.2600502028465271 (for 500 epochs)
Epoch  476  time  0.15071821212768555
Average time per epoch 0.2603516392707825 (for 500 epochs)
Epoch  477  time  0.14496254920959473
Average time per epoch 0.26064156436920166 (for 500 epochs)
Epoch  478  time  0.14283370971679688
Average time per epoch 0.26092723178863525 (for 500 epochs)
Epoch  479  time  0.14366793632507324
Average time per epoch 0.2612145676612854 (for 500 epochs)
Epoch  480  time  0.14590239524841309
Epoch  480  loss  0.14084233287142836 correct 50
Average time per epoch 0.2615063724517822 (for 500 epochs)
Epoch  481  time  0.14667320251464844
Average time per epoch 0.2617997188568115 (for 500 epochs)
Epoch  482  time  0.17017674446105957
Average time per epoch 0.26214007234573367 (for 500 epochs)
Epoch  483  time  0.15619826316833496
Average time per epoch 0.2624524688720703 (for 500 epochs)
Epoch  484  time  0.14618587493896484
Average time per epoch 0.26274484062194825 (for 500 epochs)
Epoch  485  time  0.1440715789794922
Average time per epoch 0.26303298377990725 (for 500 epochs)
Epoch  486  time  0.16185593605041504
Average time per epoch 0.26335669565200803 (for 500 epochs)
Epoch  487  time  0.17369699478149414
Average time per epoch 0.26370408964157105 (for 500 epochs)
Epoch  488  time  0.1528170108795166
Average time per epoch 0.2640097236633301 (for 500 epochs)
Epoch  489  time  0.16535377502441406
Average time per epoch 0.2643404312133789 (for 500 epochs)
Epoch  490  time  0.14800000190734863
Epoch  490  loss  0.1489842447611973 correct 50
Average time per epoch 0.2646364312171936 (for 500 epochs)
Epoch  491  time  0.13911652565002441
Average time per epoch 0.26491466426849364 (for 500 epochs)
Epoch  492  time  0.15567851066589355
Average time per epoch 0.2652260212898254 (for 500 epochs)
Epoch  493  time  0.14588594436645508
Average time per epoch 0.26551779317855834 (for 500 epochs)
Epoch  494  time  0.14978694915771484
Average time per epoch 0.26581736707687376 (for 500 epochs)
Epoch  495  time  0.14628982543945312
Average time per epoch 0.2661099467277527 (for 500 epochs)
Epoch  496  time  0.16316437721252441
Average time per epoch 0.26643627548217774 (for 500 epochs)
Epoch  497  time  0.14278054237365723
Average time per epoch 0.26672183656692505 (for 500 epochs)
Epoch  498  time  0.1505413055419922
Average time per epoch 0.267022919178009 (for 500 epochs)
Epoch  499  time  0.1483747959136963
Average time per epoch 0.2673196687698364 (for 500 epochs)
```
# CPU XOR Dataset:
```
Epoch  0  time  19.672213077545166
Epoch  0  loss  6.631631230242888 correct 34
Average time per epoch 0.03934442615509033 (for 500 epochs)
Epoch  1  time  0.14945125579833984
Average time per epoch 0.03964332866668701 (for 500 epochs)
Epoch  2  time  0.14916181564331055
Average time per epoch 0.03994165229797363 (for 500 epochs)
Epoch  3  time  0.23428034782409668
Average time per epoch 0.040410212993621825 (for 500 epochs)
Epoch  4  time  0.14000511169433594
Average time per epoch 0.0406902232170105 (for 500 epochs)
Epoch  5  time  0.14499926567077637
Average time per epoch 0.04098022174835205 (for 500 epochs)
Epoch  6  time  0.14817452430725098
Average time per epoch 0.041276570796966554 (for 500 epochs)
Epoch  7  time  0.13907575607299805
Average time per epoch 0.04155472230911255 (for 500 epochs)
Epoch  8  time  0.14885425567626953
Average time per epoch 0.04185243082046509 (for 500 epochs)
Epoch  9  time  0.13878560066223145
Average time per epoch 0.04213000202178955 (for 500 epochs)
Epoch  10  time  0.15685296058654785
Epoch  10  loss  5.215154638793276 correct 44
Average time per epoch 0.04244370794296265 (for 500 epochs)
Epoch  11  time  0.13815593719482422
Average time per epoch 0.042720019817352295 (for 500 epochs)
Epoch  12  time  0.14397025108337402
Average time per epoch 0.04300796031951904 (for 500 epochs)
Epoch  13  time  0.14406466484069824
Average time per epoch 0.04329608964920044 (for 500 epochs)
Epoch  14  time  0.14330172538757324
Average time per epoch 0.04358269309997559 (for 500 epochs)
Epoch  15  time  0.1465446949005127
Average time per epoch 0.04387578248977661 (for 500 epochs)
Epoch  16  time  0.14327669143676758
Average time per epoch 0.044162335872650144 (for 500 epochs)
Epoch  17  time  0.15267682075500488
Average time per epoch 0.04446768951416016 (for 500 epochs)
Epoch  18  time  0.1433250904083252
Average time per epoch 0.04475433969497681 (for 500 epochs)
Epoch  19  time  0.17249536514282227
Average time per epoch 0.04509933042526245 (for 500 epochs)
Epoch  20  time  0.14337396621704102
Epoch  20  loss  3.7474788223331905 correct 44
Average time per epoch 0.045386078357696535 (for 500 epochs)
Epoch  21  time  0.14135265350341797
Average time per epoch 0.045668783664703366 (for 500 epochs)
Epoch  22  time  0.15111637115478516
Average time per epoch 0.04597101640701294 (for 500 epochs)
Epoch  23  time  0.1427173614501953
Average time per epoch 0.04625645112991333 (for 500 epochs)
Epoch  24  time  0.16564488410949707
Average time per epoch 0.04658774089813232 (for 500 epochs)
Epoch  25  time  0.14293336868286133
Average time per epoch 0.04687360763549805 (for 500 epochs)
Epoch  26  time  0.14416241645812988
Average time per epoch 0.047161932468414304 (for 500 epochs)
Epoch  27  time  0.15395140647888184
Average time per epoch 0.04746983528137207 (for 500 epochs)
Epoch  28  time  0.14258050918579102
Average time per epoch 0.047754996299743654 (for 500 epochs)
Epoch  29  time  0.14603257179260254
Average time per epoch 0.048047061443328855 (for 500 epochs)
Epoch  30  time  0.13898110389709473
Epoch  30  loss  4.510018641155128 correct 47
Average time per epoch 0.048325023651123046 (for 500 epochs)
Epoch  31  time  0.16922283172607422
Average time per epoch 0.04866346931457519 (for 500 epochs)
Epoch  32  time  0.13937067985534668
Average time per epoch 0.04894221067428589 (for 500 epochs)
Epoch  33  time  0.148298978805542
Average time per epoch 0.049238808631896974 (for 500 epochs)
Epoch  34  time  0.13897943496704102
Average time per epoch 0.04951676750183105 (for 500 epochs)
Epoch  35  time  0.1521921157836914
Average time per epoch 0.04982115173339844 (for 500 epochs)
Epoch  36  time  0.14006829261779785
Average time per epoch 0.05010128831863404 (for 500 epochs)
Epoch  37  time  0.146956205368042
Average time per epoch 0.050395200729370114 (for 500 epochs)
Epoch  38  time  0.1593494415283203
Average time per epoch 0.050713899612426755 (for 500 epochs)
Epoch  39  time  0.1452012062072754
Average time per epoch 0.05100430202484131 (for 500 epochs)
Epoch  40  time  0.14324402809143066
Epoch  40  loss  2.6802837870523852 correct 48
Average time per epoch 0.05129079008102417 (for 500 epochs)
Epoch  41  time  0.15726017951965332
Average time per epoch 0.05160531044006347 (for 500 epochs)
Epoch  42  time  0.1441795825958252
Average time per epoch 0.05189366960525513 (for 500 epochs)
Epoch  43  time  0.14534378051757812
Average time per epoch 0.05218435716629028 (for 500 epochs)
Epoch  44  time  0.1383051872253418
Average time per epoch 0.05246096754074097 (for 500 epochs)
Epoch  45  time  0.15909504890441895
Average time per epoch 0.0527791576385498 (for 500 epochs)
Epoch  46  time  0.1432967185974121
Average time per epoch 0.05306575107574463 (for 500 epochs)
Epoch  47  time  0.1460115909576416
Average time per epoch 0.05335777425765991 (for 500 epochs)
Epoch  48  time  0.14505243301391602
Average time per epoch 0.05364787912368774 (for 500 epochs)
Epoch  49  time  0.14660978317260742
Average time per epoch 0.05394109869003296 (for 500 epochs)
Epoch  50  time  0.1461653709411621
Epoch  50  loss  1.9778407446821145 correct 48
Average time per epoch 0.05423342943191528 (for 500 epochs)
Epoch  51  time  0.142195463180542
Average time per epoch 0.05451782035827637 (for 500 epochs)
Epoch  52  time  0.17157435417175293
Average time per epoch 0.05486096906661987 (for 500 epochs)
Epoch  53  time  0.1381843090057373
Average time per epoch 0.055137337684631346 (for 500 epochs)
Epoch  54  time  0.14376568794250488
Average time per epoch 0.05542486906051636 (for 500 epochs)
Epoch  55  time  0.14171910285949707
Average time per epoch 0.05570830726623535 (for 500 epochs)
Epoch  56  time  0.1474311351776123
Average time per epoch 0.056003169536590576 (for 500 epochs)
Epoch  57  time  0.1394665241241455
Average time per epoch 0.056282102584838865 (for 500 epochs)
Epoch  58  time  0.14342069625854492
Average time per epoch 0.05656894397735596 (for 500 epochs)
Epoch  59  time  0.15285706520080566
Average time per epoch 0.05687465810775757 (for 500 epochs)
Epoch  60  time  0.15490317344665527
Epoch  60  loss  1.9972892177101016 correct 48
Average time per epoch 0.05718446445465088 (for 500 epochs)
Epoch  61  time  0.13790321350097656
Average time per epoch 0.057460270881652835 (for 500 epochs)
Epoch  62  time  0.1521756649017334
Average time per epoch 0.0577646222114563 (for 500 epochs)
Epoch  63  time  0.13999176025390625
Average time per epoch 0.05804460573196411 (for 500 epochs)
Epoch  64  time  0.14284396171569824
Average time per epoch 0.05833029365539551 (for 500 epochs)
Epoch  65  time  0.1398320198059082
Average time per epoch 0.05860995769500733 (for 500 epochs)
Epoch  66  time  0.16132688522338867
Average time per epoch 0.0589326114654541 (for 500 epochs)
Epoch  67  time  0.14440274238586426
Average time per epoch 0.05922141695022583 (for 500 epochs)
Epoch  68  time  0.1405010223388672
Average time per epoch 0.059502418994903564 (for 500 epochs)
Epoch  69  time  0.15640830993652344
Average time per epoch 0.05981523561477661 (for 500 epochs)
Epoch  70  time  0.14145350456237793
Epoch  70  loss  2.8045401373561782 correct 48
Average time per epoch 0.06009814262390137 (for 500 epochs)
Epoch  71  time  0.1447286605834961
Average time per epoch 0.06038759994506836 (for 500 epochs)
Epoch  72  time  0.15688204765319824
Average time per epoch 0.06070136404037475 (for 500 epochs)
Epoch  73  time  0.14401793479919434
Average time per epoch 0.060989399909973145 (for 500 epochs)
Epoch  74  time  0.14706707000732422
Average time per epoch 0.06128353404998779 (for 500 epochs)
Epoch  75  time  0.1534883975982666
Average time per epoch 0.061590510845184326 (for 500 epochs)
Epoch  76  time  0.1674637794494629
Average time per epoch 0.06192543840408325 (for 500 epochs)
Epoch  77  time  0.16135573387145996
Average time per epoch 0.06224814987182617 (for 500 epochs)
Epoch  78  time  0.1531391143798828
Average time per epoch 0.06255442810058594 (for 500 epochs)
Epoch  79  time  0.18996024131774902
Average time per epoch 0.06293434858322143 (for 500 epochs)
Epoch  80  time  0.16896486282348633
Epoch  80  loss  2.467949309759978 correct 48
Average time per epoch 0.0632722783088684 (for 500 epochs)
Epoch  81  time  0.15709662437438965
Average time per epoch 0.0635864715576172 (for 500 epochs)
Epoch  82  time  0.1548750400543213
Average time per epoch 0.06389622163772583 (for 500 epochs)
Epoch  83  time  0.14433598518371582
Average time per epoch 0.06418489360809326 (for 500 epochs)
Epoch  84  time  0.13724732398986816
Average time per epoch 0.064459388256073 (for 500 epochs)
Epoch  85  time  0.1580352783203125
Average time per epoch 0.06477545881271363 (for 500 epochs)
Epoch  86  time  0.1475064754486084
Average time per epoch 0.06507047176361085 (for 500 epochs)
Epoch  87  time  0.15061640739440918
Average time per epoch 0.06537170457839966 (for 500 epochs)
Epoch  88  time  0.1436910629272461
Average time per epoch 0.06565908670425415 (for 500 epochs)
Epoch  89  time  0.15276694297790527
Average time per epoch 0.06596462059020997 (for 500 epochs)
Epoch  90  time  0.13840103149414062
Epoch  90  loss  1.6149546746219814 correct 49
Average time per epoch 0.06624142265319824 (for 500 epochs)
Epoch  91  time  0.14492249488830566
Average time per epoch 0.06653126764297486 (for 500 epochs)
Epoch  92  time  0.1582472324371338
Average time per epoch 0.06684776210784912 (for 500 epochs)
Epoch  93  time  0.1450052261352539
Average time per epoch 0.06713777256011963 (for 500 epochs)
Epoch  94  time  0.14402294158935547
Average time per epoch 0.06742581844329834 (for 500 epochs)
Epoch  95  time  0.1468486785888672
Average time per epoch 0.06771951580047607 (for 500 epochs)
Epoch  96  time  0.1477212905883789
Average time per epoch 0.06801495838165283 (for 500 epochs)
Epoch  97  time  0.14181208610534668
Average time per epoch 0.06829858255386352 (for 500 epochs)
Epoch  98  time  0.13905978202819824
Average time per epoch 0.06857670211791993 (for 500 epochs)
Epoch  99  time  0.1555955410003662
Average time per epoch 0.06888789319992066 (for 500 epochs)
Epoch  100  time  0.14437055587768555
Epoch  100  loss  0.5815045360367251 correct 49
Average time per epoch 0.06917663431167602 (for 500 epochs)
Epoch  101  time  0.15116453170776367
Average time per epoch 0.06947896337509155 (for 500 epochs)
Epoch  102  time  0.1407947540283203
Average time per epoch 0.0697605528831482 (for 500 epochs)
Epoch  103  time  0.14316415786743164
Average time per epoch 0.07004688119888305 (for 500 epochs)
Epoch  104  time  0.1457383632659912
Average time per epoch 0.07033835792541504 (for 500 epochs)
Epoch  105  time  0.14433717727661133
Average time per epoch 0.07062703227996826 (for 500 epochs)
Epoch  106  time  0.1684267520904541
Average time per epoch 0.07096388578414917 (for 500 epochs)
Epoch  107  time  0.14810538291931152
Average time per epoch 0.07126009654998779 (for 500 epochs)
Epoch  108  time  0.15096426010131836
Average time per epoch 0.07156202507019042 (for 500 epochs)
Epoch  109  time  0.14035367965698242
Average time per epoch 0.0718427324295044 (for 500 epochs)
Epoch  110  time  0.14941024780273438
Epoch  110  loss  1.891210120075547 correct 49
Average time per epoch 0.07214155292510986 (for 500 epochs)
Epoch  111  time  0.13906574249267578
Average time per epoch 0.07241968441009522 (for 500 epochs)
Epoch  112  time  0.1468362808227539
Average time per epoch 0.07271335697174072 (for 500 epochs)
Epoch  113  time  0.1531219482421875
Average time per epoch 0.07301960086822509 (for 500 epochs)
Epoch  114  time  0.14456701278686523
Average time per epoch 0.07330873489379883 (for 500 epochs)
Epoch  115  time  0.1412675380706787
Average time per epoch 0.07359126996994018 (for 500 epochs)
Epoch  116  time  0.14296483993530273
Average time per epoch 0.07387719964981079 (for 500 epochs)
Epoch  117  time  0.15320873260498047
Average time per epoch 0.07418361711502075 (for 500 epochs)
Epoch  118  time  0.14975333213806152
Average time per epoch 0.07448312377929688 (for 500 epochs)
Epoch  119  time  0.13918590545654297
Average time per epoch 0.07476149559020996 (for 500 epochs)
Epoch  120  time  0.16518402099609375
Epoch  120  loss  0.4519312476120103 correct 50
Average time per epoch 0.07509186363220215 (for 500 epochs)
Epoch  121  time  0.14601516723632812
Average time per epoch 0.0753838939666748 (for 500 epochs)
Epoch  122  time  0.15071415901184082
Average time per epoch 0.07568532228469849 (for 500 epochs)
Epoch  123  time  0.14792132377624512
Average time per epoch 0.07598116493225097 (for 500 epochs)
Epoch  124  time  0.14638924598693848
Average time per epoch 0.07627394342422486 (for 500 epochs)
Epoch  125  time  0.14826583862304688
Average time per epoch 0.07657047510147094 (for 500 epochs)
Epoch  126  time  0.14752197265625
Average time per epoch 0.07686551904678345 (for 500 epochs)
Epoch  127  time  0.16144633293151855
Average time per epoch 0.07718841171264648 (for 500 epochs)
Epoch  128  time  0.1443498134613037
Average time per epoch 0.0774771113395691 (for 500 epochs)
Epoch  129  time  0.15587282180786133
Average time per epoch 0.07778885698318481 (for 500 epochs)
Epoch  130  time  0.14650511741638184
Epoch  130  loss  1.730751652945964 correct 49
Average time per epoch 0.07808186721801758 (for 500 epochs)
Epoch  131  time  0.15462064743041992
Average time per epoch 0.07839110851287842 (for 500 epochs)
Epoch  132  time  0.15512752532958984
Average time per epoch 0.0787013635635376 (for 500 epochs)
Epoch  133  time  0.16290712356567383
Average time per epoch 0.07902717781066894 (for 500 epochs)
Epoch  134  time  0.15672540664672852
Average time per epoch 0.0793406286239624 (for 500 epochs)
Epoch  135  time  0.14720559120178223
Average time per epoch 0.07963503980636596 (for 500 epochs)
Epoch  136  time  0.1527559757232666
Average time per epoch 0.0799405517578125 (for 500 epochs)
Epoch  137  time  0.14822840690612793
Average time per epoch 0.08023700857162476 (for 500 epochs)
Epoch  138  time  0.15646767616271973
Average time per epoch 0.08054994392395019 (for 500 epochs)
Epoch  139  time  0.14776301383972168
Average time per epoch 0.08084546995162964 (for 500 epochs)
Epoch  140  time  0.16553258895874023
Epoch  140  loss  0.6286596626346629 correct 50
Average time per epoch 0.08117653512954712 (for 500 epochs)
Epoch  141  time  0.1509401798248291
Average time per epoch 0.08147841548919678 (for 500 epochs)
Epoch  142  time  0.1470625400543213
Average time per epoch 0.08177254056930541 (for 500 epochs)
Epoch  143  time  0.14016962051391602
Average time per epoch 0.08205287981033325 (for 500 epochs)
Epoch  144  time  0.1458733081817627
Average time per epoch 0.08234462642669678 (for 500 epochs)
Epoch  145  time  0.14538192749023438
Average time per epoch 0.08263539028167724 (for 500 epochs)
Epoch  146  time  0.14751935005187988
Average time per epoch 0.082930428981781 (for 500 epochs)
Epoch  147  time  0.1546950340270996
Average time per epoch 0.08323981904983521 (for 500 epochs)
Epoch  148  time  0.15259027481079102
Average time per epoch 0.08354499959945678 (for 500 epochs)
Epoch  149  time  0.15079998970031738
Average time per epoch 0.08384659957885743 (for 500 epochs)
Epoch  150  time  0.1473090648651123
Epoch  150  loss  1.3564151765801746 correct 49
Average time per epoch 0.08414121770858765 (for 500 epochs)
Epoch  151  time  0.14828872680664062
Average time per epoch 0.08443779516220093 (for 500 epochs)
Epoch  152  time  0.14827752113342285
Average time per epoch 0.08473435020446778 (for 500 epochs)
Epoch  153  time  0.1474747657775879
Average time per epoch 0.08502929973602295 (for 500 epochs)
Epoch  154  time  0.1675856113433838
Average time per epoch 0.08536447095870972 (for 500 epochs)
Epoch  155  time  0.147629976272583
Average time per epoch 0.08565973091125488 (for 500 epochs)
Epoch  156  time  0.14936184883117676
Average time per epoch 0.08595845460891724 (for 500 epochs)
Epoch  157  time  0.14117956161499023
Average time per epoch 0.08624081373214722 (for 500 epochs)
Epoch  158  time  0.14687252044677734
Average time per epoch 0.08653455877304077 (for 500 epochs)
Epoch  159  time  0.1490650177001953
Average time per epoch 0.08683268880844117 (for 500 epochs)
Epoch  160  time  0.16192626953125
Epoch  160  loss  0.7898172272688404 correct 49
Average time per epoch 0.08715654134750367 (for 500 epochs)
Epoch  161  time  0.14547038078308105
Average time per epoch 0.08744748210906983 (for 500 epochs)
Epoch  162  time  0.15392327308654785
Average time per epoch 0.08775532865524292 (for 500 epochs)
Epoch  163  time  0.14383244514465332
Average time per epoch 0.08804299354553223 (for 500 epochs)
Epoch  164  time  0.14552783966064453
Average time per epoch 0.08833404922485352 (for 500 epochs)
Epoch  165  time  0.14792799949645996
Average time per epoch 0.08862990522384644 (for 500 epochs)
Epoch  166  time  0.1486341953277588
Average time per epoch 0.08892717361450195 (for 500 epochs)
Epoch  167  time  0.1589803695678711
Average time per epoch 0.08924513435363769 (for 500 epochs)
Epoch  168  time  0.1469120979309082
Average time per epoch 0.08953895854949952 (for 500 epochs)
Epoch  169  time  0.1418018341064453
Average time per epoch 0.0898225622177124 (for 500 epochs)
Epoch  170  time  0.14451026916503906
Epoch  170  loss  0.4848296738956391 correct 49
Average time per epoch 0.09011158275604247 (for 500 epochs)
Epoch  171  time  0.14359068870544434
Average time per epoch 0.09039876413345337 (for 500 epochs)
Epoch  172  time  0.14751696586608887
Average time per epoch 0.09069379806518554 (for 500 epochs)
Epoch  173  time  0.1475832462310791
Average time per epoch 0.09098896455764771 (for 500 epochs)
Epoch  174  time  0.162977933883667
Average time per epoch 0.09131492042541504 (for 500 epochs)
Epoch  175  time  0.14038491249084473
Average time per epoch 0.09159569025039672 (for 500 epochs)
Epoch  176  time  0.14946603775024414
Average time per epoch 0.09189462232589722 (for 500 epochs)
Epoch  177  time  0.1483011245727539
Average time per epoch 0.09219122457504272 (for 500 epochs)
Epoch  178  time  0.14748525619506836
Average time per epoch 0.09248619508743286 (for 500 epochs)
Epoch  179  time  0.14561676979064941
Average time per epoch 0.09277742862701416 (for 500 epochs)
Epoch  180  time  0.1460413932800293
Epoch  180  loss  0.7143350502464981 correct 49
Average time per epoch 0.09306951141357422 (for 500 epochs)
Epoch  181  time  0.16330194473266602
Average time per epoch 0.09339611530303955 (for 500 epochs)
Epoch  182  time  0.1474621295928955
Average time per epoch 0.09369103956222534 (for 500 epochs)
Epoch  183  time  0.14654231071472168
Average time per epoch 0.09398412418365479 (for 500 epochs)
Epoch  184  time  0.1501162052154541
Average time per epoch 0.0942843565940857 (for 500 epochs)
Epoch  185  time  0.14478516578674316
Average time per epoch 0.09457392692565918 (for 500 epochs)
Epoch  186  time  0.14809536933898926
Average time per epoch 0.09487011766433716 (for 500 epochs)
Epoch  187  time  0.1463782787322998
Average time per epoch 0.09516287422180175 (for 500 epochs)
Epoch  188  time  0.15780258178710938
Average time per epoch 0.09547847938537597 (for 500 epochs)
Epoch  189  time  0.14242768287658691
Average time per epoch 0.09576333475112915 (for 500 epochs)
Epoch  190  time  0.15324878692626953
Epoch  190  loss  0.9636321765170532 correct 49
Average time per epoch 0.09606983232498169 (for 500 epochs)
Epoch  191  time  0.14468979835510254
Average time per epoch 0.09635921192169189 (for 500 epochs)
Epoch  192  time  0.1484835147857666
Average time per epoch 0.09665617895126342 (for 500 epochs)
Epoch  193  time  0.15041804313659668
Average time per epoch 0.09695701503753662 (for 500 epochs)
Epoch  194  time  0.16290068626403809
Average time per epoch 0.0972828164100647 (for 500 epochs)
Epoch  195  time  0.1435074806213379
Average time per epoch 0.09756983137130737 (for 500 epochs)
Epoch  196  time  0.14782333374023438
Average time per epoch 0.09786547803878784 (for 500 epochs)
Epoch  197  time  0.14748287200927734
Average time per epoch 0.0981604437828064 (for 500 epochs)
Epoch  198  time  0.1460416316986084
Average time per epoch 0.09845252704620361 (for 500 epochs)
Epoch  199  time  0.14959168434143066
Average time per epoch 0.09875171041488648 (for 500 epochs)
Epoch  200  time  0.1533651351928711
Epoch  200  loss  0.811260741364843 correct 50
Average time per epoch 0.09905844068527221 (for 500 epochs)
Epoch  201  time  0.16608500480651855
Average time per epoch 0.09939061069488525 (for 500 epochs)
Epoch  202  time  0.1462545394897461
Average time per epoch 0.09968311977386475 (for 500 epochs)
Epoch  203  time  0.14856719970703125
Average time per epoch 0.09998025417327881 (for 500 epochs)
Epoch  204  time  0.14750361442565918
Average time per epoch 0.10027526140213013 (for 500 epochs)
Epoch  205  time  0.14612412452697754
Average time per epoch 0.10056750965118408 (for 500 epochs)
Epoch  206  time  0.15217804908752441
Average time per epoch 0.10087186574935914 (for 500 epochs)
Epoch  207  time  0.1589338779449463
Average time per epoch 0.10118973350524903 (for 500 epochs)
Epoch  208  time  0.1653749942779541
Average time per epoch 0.10152048349380494 (for 500 epochs)
Epoch  209  time  0.14539408683776855
Average time per epoch 0.10181127166748047 (for 500 epochs)
Epoch  210  time  0.1468040943145752
Epoch  210  loss  0.5051428652361635 correct 49
Average time per epoch 0.10210487985610962 (for 500 epochs)
Epoch  211  time  0.14496898651123047
Average time per epoch 0.10239481782913208 (for 500 epochs)
Epoch  212  time  0.14909625053405762
Average time per epoch 0.1026930103302002 (for 500 epochs)
Epoch  213  time  0.14626002311706543
Average time per epoch 0.10298553037643432 (for 500 epochs)
Epoch  214  time  0.14983296394348145
Average time per epoch 0.10328519630432129 (for 500 epochs)
Epoch  215  time  0.15836596488952637
Average time per epoch 0.10360192823410035 (for 500 epochs)
Epoch  216  time  0.1569957733154297
Average time per epoch 0.1039159197807312 (for 500 epochs)
Epoch  217  time  0.1466965675354004
Average time per epoch 0.104209312915802 (for 500 epochs)
Epoch  218  time  0.1541459560394287
Average time per epoch 0.10451760482788086 (for 500 epochs)
Epoch  219  time  0.1468980312347412
Average time per epoch 0.10481140089035035 (for 500 epochs)
Epoch  220  time  0.1475532054901123
Epoch  220  loss  0.6511505580523862 correct 49
Average time per epoch 0.10510650730133056 (for 500 epochs)
Epoch  221  time  0.1601879596710205
Average time per epoch 0.1054268832206726 (for 500 epochs)
Epoch  222  time  0.14759159088134766
Average time per epoch 0.1057220664024353 (for 500 epochs)
Epoch  223  time  0.15244150161743164
Average time per epoch 0.10602694940567016 (for 500 epochs)
Epoch  224  time  0.14257264137268066
Average time per epoch 0.10631209468841553 (for 500 epochs)
Epoch  225  time  0.15263748168945312
Average time per epoch 0.10661736965179443 (for 500 epochs)
Epoch  226  time  0.14125943183898926
Average time per epoch 0.10689988851547241 (for 500 epochs)
Epoch  227  time  0.14951610565185547
Average time per epoch 0.10719892072677613 (for 500 epochs)
Epoch  228  time  0.15931010246276855
Average time per epoch 0.10751754093170166 (for 500 epochs)
Epoch  229  time  0.1532883644104004
Average time per epoch 0.10782411766052247 (for 500 epochs)
Epoch  230  time  0.14760565757751465
Epoch  230  loss  0.03439247019957512 correct 49
Average time per epoch 0.10811932897567748 (for 500 epochs)
Epoch  231  time  0.15274596214294434
Average time per epoch 0.10842482089996337 (for 500 epochs)
Epoch  232  time  0.14440202713012695
Average time per epoch 0.10871362495422364 (for 500 epochs)
Epoch  233  time  0.14433741569519043
Average time per epoch 0.10900229978561402 (for 500 epochs)
Epoch  234  time  0.14206647872924805
Average time per epoch 0.10928643274307251 (for 500 epochs)
Epoch  235  time  0.16685771942138672
Average time per epoch 0.10962014818191529 (for 500 epochs)
Epoch  236  time  0.15149164199829102
Average time per epoch 0.10992313146591187 (for 500 epochs)
Epoch  237  time  0.14653992652893066
Average time per epoch 0.11021621131896972 (for 500 epochs)
Epoch  238  time  0.14635777473449707
Average time per epoch 0.11050892686843872 (for 500 epochs)
Epoch  239  time  0.15149521827697754
Average time per epoch 0.11081191730499268 (for 500 epochs)
Epoch  240  time  0.13900303840637207
Epoch  240  loss  1.4381612042108511 correct 50
Average time per epoch 0.11108992338180541 (for 500 epochs)
Epoch  241  time  0.14298057556152344
Average time per epoch 0.11137588453292847 (for 500 epochs)
Epoch  242  time  0.15701603889465332
Average time per epoch 0.11168991661071777 (for 500 epochs)
Epoch  243  time  0.1470785140991211
Average time per epoch 0.11198407363891602 (for 500 epochs)
Epoch  244  time  0.13729429244995117
Average time per epoch 0.11225866222381592 (for 500 epochs)
Epoch  245  time  0.15011262893676758
Average time per epoch 0.11255888748168945 (for 500 epochs)
Epoch  246  time  0.13982915878295898
Average time per epoch 0.11283854579925537 (for 500 epochs)
Epoch  247  time  0.15062713623046875
Average time per epoch 0.11313980007171631 (for 500 epochs)
Epoch  248  time  0.13867616653442383
Average time per epoch 0.11341715240478516 (for 500 epochs)
Epoch  249  time  0.15602684020996094
Average time per epoch 0.11372920608520508 (for 500 epochs)
Epoch  250  time  0.155287504196167
Epoch  250  loss  1.1235635995504396 correct 49
Average time per epoch 0.11403978109359741 (for 500 epochs)
Epoch  251  time  0.15001583099365234
Average time per epoch 0.11433981275558472 (for 500 epochs)
Epoch  252  time  0.14530229568481445
Average time per epoch 0.11463041734695435 (for 500 epochs)
Epoch  253  time  0.1399369239807129
Average time per epoch 0.11491029119491578 (for 500 epochs)
Epoch  254  time  0.1497058868408203
Average time per epoch 0.11520970296859741 (for 500 epochs)
Epoch  255  time  0.14109444618225098
Average time per epoch 0.11549189186096191 (for 500 epochs)
Epoch  256  time  0.15735816955566406
Average time per epoch 0.11580660820007324 (for 500 epochs)
Epoch  257  time  0.1457839012145996
Average time per epoch 0.11609817600250244 (for 500 epochs)
Epoch  258  time  0.1465456485748291
Average time per epoch 0.1163912672996521 (for 500 epochs)
Epoch  259  time  0.1454780101776123
Average time per epoch 0.11668222332000733 (for 500 epochs)
Epoch  260  time  0.14448165893554688
Epoch  260  loss  0.2289384479332333 correct 49
Average time per epoch 0.11697118663787842 (for 500 epochs)
Epoch  261  time  0.1484990119934082
Average time per epoch 0.11726818466186524 (for 500 epochs)
Epoch  262  time  0.15739083290100098
Average time per epoch 0.11758296632766724 (for 500 epochs)
Epoch  263  time  0.1465773582458496
Average time per epoch 0.11787612104415894 (for 500 epochs)
Epoch  264  time  0.14511418342590332
Average time per epoch 0.11816634941101074 (for 500 epochs)
Epoch  265  time  0.14870524406433105
Average time per epoch 0.1184637598991394 (for 500 epochs)
Epoch  266  time  0.15596723556518555
Average time per epoch 0.11877569437026977 (for 500 epochs)
Epoch  267  time  0.14773106575012207
Average time per epoch 0.11907115650177003 (for 500 epochs)
Epoch  268  time  0.1463184356689453
Average time per epoch 0.11936379337310792 (for 500 epochs)
Epoch  269  time  0.15807318687438965
Average time per epoch 0.11967993974685669 (for 500 epochs)
Epoch  270  time  0.14510703086853027
Epoch  270  loss  1.2615681128948049 correct 50
Average time per epoch 0.11997015380859374 (for 500 epochs)
Epoch  271  time  0.14705801010131836
Average time per epoch 0.12026426982879639 (for 500 epochs)
Epoch  272  time  0.1470479965209961
Average time per epoch 0.12055836582183838 (for 500 epochs)
Epoch  273  time  0.14805006980895996
Average time per epoch 0.1208544659614563 (for 500 epochs)
Epoch  274  time  0.1462407112121582
Average time per epoch 0.12114694738388061 (for 500 epochs)
Epoch  275  time  0.14889121055603027
Average time per epoch 0.12144472980499267 (for 500 epochs)
Epoch  276  time  0.15920233726501465
Average time per epoch 0.12176313447952271 (for 500 epochs)
Epoch  277  time  0.14390993118286133
Average time per epoch 0.12205095434188842 (for 500 epochs)
Epoch  278  time  0.14274191856384277
Average time per epoch 0.12233643817901611 (for 500 epochs)
Epoch  279  time  0.14307713508605957
Average time per epoch 0.12262259244918823 (for 500 epochs)
Epoch  280  time  0.15659046173095703
Epoch  280  loss  0.8716651593321252 correct 49
Average time per epoch 0.12293577337265015 (for 500 epochs)
Epoch  281  time  0.14912819862365723
Average time per epoch 0.12323402976989746 (for 500 epochs)
Epoch  282  time  0.14511775970458984
Average time per epoch 0.12352426528930664 (for 500 epochs)
Epoch  283  time  0.1673264503479004
Average time per epoch 0.12385891819000244 (for 500 epochs)
Epoch  284  time  0.16045522689819336
Average time per epoch 0.12417982864379883 (for 500 epochs)
Epoch  285  time  0.14265990257263184
Average time per epoch 0.12446514844894409 (for 500 epochs)
Epoch  286  time  0.14644861221313477
Average time per epoch 0.12475804567337036 (for 500 epochs)
Epoch  287  time  0.14934277534484863
Average time per epoch 0.12505673122406005 (for 500 epochs)
Epoch  288  time  0.15312457084655762
Average time per epoch 0.12536298036575316 (for 500 epochs)
Epoch  289  time  0.14592719078063965
Average time per epoch 0.12565483474731445 (for 500 epochs)
Epoch  290  time  0.16112852096557617
Epoch  290  loss  0.8717600425081421 correct 50
Average time per epoch 0.1259770917892456 (for 500 epochs)
Epoch  291  time  0.14384961128234863
Average time per epoch 0.1262647910118103 (for 500 epochs)
Epoch  292  time  0.15305852890014648
Average time per epoch 0.1265709080696106 (for 500 epochs)
Epoch  293  time  0.14627456665039062
Average time per epoch 0.12686345720291137 (for 500 epochs)
Epoch  294  time  0.18149805068969727
Average time per epoch 0.12722645330429078 (for 500 epochs)
Epoch  295  time  0.15952372550964355
Average time per epoch 0.12754550075531007 (for 500 epochs)
Epoch  296  time  0.17256426811218262
Average time per epoch 0.1278906292915344 (for 500 epochs)
Epoch  297  time  0.16082262992858887
Average time per epoch 0.12821227455139161 (for 500 epochs)
Epoch  298  time  0.15761995315551758
Average time per epoch 0.12852751445770264 (for 500 epochs)
Epoch  299  time  0.14876699447631836
Average time per epoch 0.12882504844665527 (for 500 epochs)
Epoch  300  time  0.15207576751708984
Epoch  300  loss  0.38921192310735553 correct 49
Average time per epoch 0.12912919998168945 (for 500 epochs)
Epoch  301  time  0.1473097801208496
Average time per epoch 0.12942381954193116 (for 500 epochs)
Epoch  302  time  0.14974713325500488
Average time per epoch 0.12972331380844115 (for 500 epochs)
Epoch  303  time  0.1542518138885498
Average time per epoch 0.13003181743621825 (for 500 epochs)
Epoch  304  time  0.15894198417663574
Average time per epoch 0.13034970140457153 (for 500 epochs)
Epoch  305  time  0.26373839378356934
Average time per epoch 0.13087717819213868 (for 500 epochs)
Epoch  306  time  1.6300089359283447
Average time per epoch 0.13413719606399535 (for 500 epochs)
Epoch  307  time  0.6810324192047119
Average time per epoch 0.13549926090240477 (for 500 epochs)
Epoch  308  time  0.3689901828765869
Average time per epoch 0.13623724126815795 (for 500 epochs)
Epoch  309  time  1.145132303237915
Average time per epoch 0.13852750587463378 (for 500 epochs)
Epoch  310  time  0.49375176429748535
Epoch  310  loss  0.19717054502293263 correct 49
Average time per epoch 0.13951500940322875 (for 500 epochs)
Epoch  311  time  0.8325440883636475
Average time per epoch 0.14118009757995606 (for 500 epochs)
Epoch  312  time  0.37805843353271484
Average time per epoch 0.14193621444702148 (for 500 epochs)
Epoch  313  time  0.3701496124267578
Average time per epoch 0.142676513671875 (for 500 epochs)
Epoch  314  time  1.205887794494629
Average time per epoch 0.14508828926086426 (for 500 epochs)
Epoch  315  time  2.485942840576172
Average time per epoch 0.1500601749420166 (for 500 epochs)
Epoch  316  time  1.7166051864624023
Average time per epoch 0.15349338531494142 (for 500 epochs)
Epoch  317  time  0.14149713516235352
Average time per epoch 0.1537763795852661 (for 500 epochs)
Epoch  318  time  0.14719343185424805
Average time per epoch 0.1540707664489746 (for 500 epochs)
Epoch  319  time  0.15156936645507812
Average time per epoch 0.15437390518188476 (for 500 epochs)
Epoch  320  time  0.14695143699645996
Epoch  320  loss  0.19759323468497172 correct 49
Average time per epoch 0.15466780805587768 (for 500 epochs)
Epoch  321  time  0.14793682098388672
Average time per epoch 0.15496368169784547 (for 500 epochs)
Epoch  322  time  0.1439676284790039
Average time per epoch 0.15525161695480347 (for 500 epochs)
Epoch  323  time  0.14194726943969727
Average time per epoch 0.15553551149368286 (for 500 epochs)
Epoch  324  time  0.14594292640686035
Average time per epoch 0.1558273973464966 (for 500 epochs)
Epoch  325  time  0.17255592346191406
Average time per epoch 0.1561725091934204 (for 500 epochs)
Epoch  326  time  0.1419687271118164
Average time per epoch 0.15645644664764405 (for 500 epochs)
Epoch  327  time  0.14191126823425293
Average time per epoch 0.15674026918411255 (for 500 epochs)
Epoch  328  time  0.15765929222106934
Average time per epoch 0.1570555877685547 (for 500 epochs)
Epoch  329  time  0.146956205368042
Average time per epoch 0.15734950017929078 (for 500 epochs)
Epoch  330  time  0.15185976028442383
Epoch  330  loss  0.6592758864507274 correct 49
Average time per epoch 0.15765321969985963 (for 500 epochs)
Epoch  331  time  0.14351320266723633
Average time per epoch 0.15794024610519408 (for 500 epochs)
Epoch  332  time  0.22349214553833008
Average time per epoch 0.15838723039627076 (for 500 epochs)
Epoch  333  time  0.16029071807861328
Average time per epoch 0.158707811832428 (for 500 epochs)
Epoch  334  time  0.14560627937316895
Average time per epoch 0.1589990243911743 (for 500 epochs)
Epoch  335  time  0.144012451171875
Average time per epoch 0.15928704929351806 (for 500 epochs)
Epoch  336  time  0.1498415470123291
Average time per epoch 0.15958673238754273 (for 500 epochs)
Epoch  337  time  0.1427769660949707
Average time per epoch 0.15987228631973266 (for 500 epochs)
Epoch  338  time  0.15219807624816895
Average time per epoch 0.160176682472229 (for 500 epochs)
Epoch  339  time  0.1656482219696045
Average time per epoch 0.16050797891616822 (for 500 epochs)
Epoch  340  time  0.15057802200317383
Epoch  340  loss  1.4027839243905957 correct 50
Average time per epoch 0.16080913496017457 (for 500 epochs)
Epoch  341  time  0.1459507942199707
Average time per epoch 0.1611010365486145 (for 500 epochs)
Epoch  342  time  0.14494848251342773
Average time per epoch 0.16139093351364137 (for 500 epochs)
Epoch  343  time  0.14595508575439453
Average time per epoch 0.16168284368515015 (for 500 epochs)
Epoch  344  time  0.14466643333435059
Average time per epoch 0.16197217655181884 (for 500 epochs)
Epoch  345  time  0.1539158821105957
Average time per epoch 0.16228000831604003 (for 500 epochs)
Epoch  346  time  0.14660859107971191
Average time per epoch 0.16257322549819947 (for 500 epochs)
Epoch  347  time  0.14292120933532715
Average time per epoch 0.1628590679168701 (for 500 epochs)
Epoch  348  time  0.14023423194885254
Average time per epoch 0.16313953638076784 (for 500 epochs)
Epoch  349  time  0.15427875518798828
Average time per epoch 0.1634480938911438 (for 500 epochs)
Epoch  350  time  0.14334845542907715
Epoch  350  loss  0.12478372669986855 correct 50
Average time per epoch 0.16373479080200196 (for 500 epochs)
Epoch  351  time  0.14563488960266113
Average time per epoch 0.16402606058120728 (for 500 epochs)
Epoch  352  time  0.15949320793151855
Average time per epoch 0.1643450469970703 (for 500 epochs)
Epoch  353  time  0.14972972869873047
Average time per epoch 0.16464450645446776 (for 500 epochs)
Epoch  354  time  0.14179372787475586
Average time per epoch 0.1649280939102173 (for 500 epochs)
Epoch  355  time  0.14703822135925293
Average time per epoch 0.1652221703529358 (for 500 epochs)
Epoch  356  time  0.13982748985290527
Average time per epoch 0.1655018253326416 (for 500 epochs)
Epoch  357  time  0.1578512191772461
Average time per epoch 0.1658175277709961 (for 500 epochs)
Epoch  358  time  0.14825725555419922
Average time per epoch 0.1661140422821045 (for 500 epochs)
Epoch  359  time  0.157073974609375
Average time per epoch 0.16642819023132324 (for 500 epochs)
Epoch  360  time  0.14182329177856445
Epoch  360  loss  1.3561482255430595 correct 50
Average time per epoch 0.16671183681488036 (for 500 epochs)
Epoch  361  time  0.1460120677947998
Average time per epoch 0.16700386095046998 (for 500 epochs)
Epoch  362  time  0.14142465591430664
Average time per epoch 0.16728671026229858 (for 500 epochs)
Epoch  363  time  0.14716649055480957
Average time per epoch 0.1675810432434082 (for 500 epochs)
Epoch  364  time  0.14938092231750488
Average time per epoch 0.1678798050880432 (for 500 epochs)
Epoch  365  time  0.1473848819732666
Average time per epoch 0.16817457485198975 (for 500 epochs)
Epoch  366  time  0.15526747703552246
Average time per epoch 0.1684851098060608 (for 500 epochs)
Epoch  367  time  0.14614582061767578
Average time per epoch 0.16877740144729614 (for 500 epochs)
Epoch  368  time  0.14121460914611816
Average time per epoch 0.16905983066558838 (for 500 epochs)
Epoch  369  time  0.1520371437072754
Average time per epoch 0.16936390495300294 (for 500 epochs)
Epoch  370  time  0.14416861534118652
Epoch  370  loss  0.7269252814491756 correct 50
Average time per epoch 0.1696522421836853 (for 500 epochs)
Epoch  371  time  0.14867758750915527
Average time per epoch 0.1699495973587036 (for 500 epochs)
Epoch  372  time  0.13998627662658691
Average time per epoch 0.17022956991195679 (for 500 epochs)
Epoch  373  time  0.16484856605529785
Average time per epoch 0.17055926704406738 (for 500 epochs)
Epoch  374  time  0.14000892639160156
Average time per epoch 0.17083928489685057 (for 500 epochs)
Epoch  375  time  0.16057634353637695
Average time per epoch 0.17116043758392335 (for 500 epochs)
Epoch  376  time  0.14671087265014648
Average time per epoch 0.17145385932922363 (for 500 epochs)
Epoch  377  time  0.1539745330810547
Average time per epoch 0.17176180839538574 (for 500 epochs)
Epoch  378  time  0.14893364906311035
Average time per epoch 0.17205967569351197 (for 500 epochs)
Epoch  379  time  0.15111494064331055
Average time per epoch 0.17236190557479858 (for 500 epochs)
Epoch  380  time  0.15674996376037598
Epoch  380  loss  0.14695130839061776 correct 49
Average time per epoch 0.17267540550231933 (for 500 epochs)
Epoch  381  time  0.15968585014343262
Average time per epoch 0.1729947772026062 (for 500 epochs)
Epoch  382  time  0.1533823013305664
Average time per epoch 0.17330154180526733 (for 500 epochs)
Epoch  383  time  0.14201807975769043
Average time per epoch 0.17358557796478272 (for 500 epochs)
Epoch  384  time  0.15231776237487793
Average time per epoch 0.17389021348953246 (for 500 epochs)
Epoch  385  time  0.1462397575378418
Average time per epoch 0.17418269300460815 (for 500 epochs)
Epoch  386  time  0.16363000869750977
Average time per epoch 0.17450995302200317 (for 500 epochs)
Epoch  387  time  0.14527201652526855
Average time per epoch 0.1748004970550537 (for 500 epochs)
Epoch  388  time  0.1483464241027832
Average time per epoch 0.1750971899032593 (for 500 epochs)
Epoch  389  time  0.15378332138061523
Average time per epoch 0.1754047565460205 (for 500 epochs)
Epoch  390  time  0.14635467529296875
Epoch  390  loss  0.20643779936127862 correct 50
Average time per epoch 0.17569746589660645 (for 500 epochs)
Epoch  391  time  0.1416773796081543
Average time per epoch 0.17598082065582277 (for 500 epochs)
Epoch  392  time  0.14917373657226562
Average time per epoch 0.1762791681289673 (for 500 epochs)
Epoch  393  time  0.15666437149047852
Average time per epoch 0.17659249687194825 (for 500 epochs)
Epoch  394  time  0.15052247047424316
Average time per epoch 0.17689354181289674 (for 500 epochs)
Epoch  395  time  0.1394205093383789
Average time per epoch 0.1771723828315735 (for 500 epochs)
Epoch  396  time  0.14200377464294434
Average time per epoch 0.17745639038085936 (for 500 epochs)
Epoch  397  time  0.14740824699401855
Average time per epoch 0.1777512068748474 (for 500 epochs)
Epoch  398  time  0.14852261543273926
Average time per epoch 0.1780482521057129 (for 500 epochs)
Epoch  399  time  0.14713430404663086
Average time per epoch 0.17834252071380616 (for 500 epochs)
Epoch  400  time  0.15471172332763672
Epoch  400  loss  0.31819303765524937 correct 49
Average time per epoch 0.17865194416046143 (for 500 epochs)
Epoch  401  time  0.14934849739074707
Average time per epoch 0.17895064115524292 (for 500 epochs)
Epoch  402  time  0.1384894847869873
Average time per epoch 0.17922762012481688 (for 500 epochs)
Epoch  403  time  0.15292859077453613
Average time per epoch 0.17953347730636596 (for 500 epochs)
Epoch  404  time  0.14594078063964844
Average time per epoch 0.17982535886764525 (for 500 epochs)
Epoch  405  time  0.16241693496704102
Average time per epoch 0.18015019273757935 (for 500 epochs)
Epoch  406  time  0.14444994926452637
Average time per epoch 0.1804390926361084 (for 500 epochs)
Epoch  407  time  0.15904521942138672
Average time per epoch 0.18075718307495117 (for 500 epochs)
Epoch  408  time  0.14481234550476074
Average time per epoch 0.1810468077659607 (for 500 epochs)
Epoch  409  time  0.14813494682312012
Average time per epoch 0.18134307765960694 (for 500 epochs)
Epoch  410  time  0.1483914852142334
Epoch  410  loss  0.2923036339376885 correct 50
Average time per epoch 0.1816398606300354 (for 500 epochs)
Epoch  411  time  0.14584898948669434
Average time per epoch 0.18193155860900878 (for 500 epochs)
Epoch  412  time  0.13983678817749023
Average time per epoch 0.18221123218536378 (for 500 epochs)
Epoch  413  time  0.1714167594909668
Average time per epoch 0.1825540657043457 (for 500 epochs)
Epoch  414  time  0.15358710289001465
Average time per epoch 0.18286123991012573 (for 500 epochs)
Epoch  415  time  0.14461445808410645
Average time per epoch 0.18315046882629393 (for 500 epochs)
Epoch  416  time  0.1387629508972168
Average time per epoch 0.18342799472808838 (for 500 epochs)
Epoch  417  time  0.14499640464782715
Average time per epoch 0.18371798753738403 (for 500 epochs)
Epoch  418  time  0.14644527435302734
Average time per epoch 0.1840108780860901 (for 500 epochs)
Epoch  419  time  0.14850425720214844
Average time per epoch 0.18430788660049438 (for 500 epochs)
Epoch  420  time  0.15663623809814453
Epoch  420  loss  0.21275042334049057 correct 49
Average time per epoch 0.18462115907669066 (for 500 epochs)
Epoch  421  time  0.14607024192810059
Average time per epoch 0.18491329956054686 (for 500 epochs)
Epoch  422  time  0.14413809776306152
Average time per epoch 0.185201575756073 (for 500 epochs)
Epoch  423  time  0.1450960636138916
Average time per epoch 0.18549176788330077 (for 500 epochs)
Epoch  424  time  0.14646553993225098
Average time per epoch 0.1857846989631653 (for 500 epochs)
Epoch  425  time  0.15106844902038574
Average time per epoch 0.18608683586120606 (for 500 epochs)
Epoch  426  time  0.14909148216247559
Average time per epoch 0.18638501882553102 (for 500 epochs)
Epoch  427  time  0.1700141429901123
Average time per epoch 0.18672504711151122 (for 500 epochs)
Epoch  428  time  0.15907788276672363
Average time per epoch 0.18704320287704468 (for 500 epochs)
Epoch  429  time  0.1604921817779541
Average time per epoch 0.18736418724060058 (for 500 epochs)
Epoch  430  time  0.15529251098632812
Epoch  430  loss  0.21147332624556064 correct 49
Average time per epoch 0.18767477226257323 (for 500 epochs)
Epoch  431  time  0.15165972709655762
Average time per epoch 0.18797809171676635 (for 500 epochs)
Epoch  432  time  0.1531364917755127
Average time per epoch 0.18828436470031737 (for 500 epochs)
Epoch  433  time  0.14912748336791992
Average time per epoch 0.18858261966705323 (for 500 epochs)
Epoch  434  time  0.15993928909301758
Average time per epoch 0.18890249824523925 (for 500 epochs)
Epoch  435  time  0.15604519844055176
Average time per epoch 0.18921458864212037 (for 500 epochs)
Epoch  436  time  0.1465306282043457
Average time per epoch 0.18950764989852906 (for 500 epochs)
Epoch  437  time  0.15671992301940918
Average time per epoch 0.18982108974456788 (for 500 epochs)
Epoch  438  time  0.1453871726989746
Average time per epoch 0.19011186408996583 (for 500 epochs)
Epoch  439  time  0.14952683448791504
Average time per epoch 0.19041091775894164 (for 500 epochs)
Epoch  440  time  0.1528337001800537
Epoch  440  loss  0.11981566504333074 correct 50
Average time per epoch 0.19071658515930176 (for 500 epochs)
Epoch  441  time  0.14754939079284668
Average time per epoch 0.19101168394088744 (for 500 epochs)
Epoch  442  time  0.1383039951324463
Average time per epoch 0.19128829193115235 (for 500 epochs)
Epoch  443  time  0.14369988441467285
Average time per epoch 0.19157569169998168 (for 500 epochs)
Epoch  444  time  0.14406800270080566
Average time per epoch 0.1918638277053833 (for 500 epochs)
Epoch  445  time  0.14401793479919434
Average time per epoch 0.19215186357498168 (for 500 epochs)
Epoch  446  time  0.1459507942199707
Average time per epoch 0.19244376516342163 (for 500 epochs)
Epoch  447  time  0.15438079833984375
Average time per epoch 0.19275252676010132 (for 500 epochs)
Epoch  448  time  0.1449127197265625
Average time per epoch 0.19304235219955446 (for 500 epochs)
Epoch  449  time  0.14293980598449707
Average time per epoch 0.19332823181152345 (for 500 epochs)
Epoch  450  time  0.14836478233337402
Epoch  450  loss  0.19489791326058914 correct 49
Average time per epoch 0.19362496137619017 (for 500 epochs)
Epoch  451  time  0.13980889320373535
Average time per epoch 0.19390457916259765 (for 500 epochs)
Epoch  452  time  0.14490938186645508
Average time per epoch 0.19419439792633056 (for 500 epochs)
Epoch  453  time  0.13990330696105957
Average time per epoch 0.1944742045402527 (for 500 epochs)
Epoch  454  time  0.16257166862487793
Average time per epoch 0.19479934787750244 (for 500 epochs)
Epoch  455  time  0.13999533653259277
Average time per epoch 0.19507933855056764 (for 500 epochs)
Epoch  456  time  0.14208078384399414
Average time per epoch 0.19536350011825562 (for 500 epochs)
Epoch  457  time  0.1545577049255371
Average time per epoch 0.1956726155281067 (for 500 epochs)
Epoch  458  time  0.1422104835510254
Average time per epoch 0.19595703649520874 (for 500 epochs)
Epoch  459  time  0.14599919319152832
Average time per epoch 0.1962490348815918 (for 500 epochs)
Epoch  460  time  0.14002370834350586
Epoch  460  loss  0.4289922716458913 correct 50
Average time per epoch 0.1965290822982788 (for 500 epochs)
Epoch  461  time  0.16133666038513184
Average time per epoch 0.19685175561904908 (for 500 epochs)
Epoch  462  time  0.1416482925415039
Average time per epoch 0.19713505220413208 (for 500 epochs)
Epoch  463  time  0.1489882469177246
Average time per epoch 0.19743302869796753 (for 500 epochs)
Epoch  464  time  0.14867424964904785
Average time per epoch 0.19773037719726563 (for 500 epochs)
Epoch  465  time  0.15189909934997559
Average time per epoch 0.19803417539596557 (for 500 epochs)
Epoch  466  time  0.14302587509155273
Average time per epoch 0.1983202271461487 (for 500 epochs)
Epoch  467  time  0.15525007247924805
Average time per epoch 0.19863072729110717 (for 500 epochs)
Epoch  468  time  0.1639566421508789
Average time per epoch 0.19895864057540893 (for 500 epochs)
Epoch  469  time  0.15066266059875488
Average time per epoch 0.19925996589660644 (for 500 epochs)
Epoch  470  time  0.14015483856201172
Epoch  470  loss  0.9745966869839958 correct 50
Average time per epoch 0.19954027557373047 (for 500 epochs)
Epoch  471  time  0.15082287788391113
Average time per epoch 0.1998419213294983 (for 500 epochs)
Epoch  472  time  0.14502882957458496
Average time per epoch 0.20013197898864746 (for 500 epochs)
Epoch  473  time  0.1463630199432373
Average time per epoch 0.20042470502853393 (for 500 epochs)
Epoch  474  time  0.1434640884399414
Average time per epoch 0.20071163320541383 (for 500 epochs)
Epoch  475  time  0.1651008129119873
Average time per epoch 0.2010418348312378 (for 500 epochs)
Epoch  476  time  0.14220690727233887
Average time per epoch 0.20132624864578247 (for 500 epochs)
Epoch  477  time  0.15228796005249023
Average time per epoch 0.20163082456588746 (for 500 epochs)
Epoch  478  time  0.14756083488464355
Average time per epoch 0.20192594623565674 (for 500 epochs)
Epoch  479  time  0.14649653434753418
Average time per epoch 0.2022189393043518 (for 500 epochs)
Epoch  480  time  0.13794755935668945
Epoch  480  loss  0.12270010357200879 correct 50
Average time per epoch 0.20249483442306518 (for 500 epochs)
Epoch  481  time  0.14991044998168945
Average time per epoch 0.20279465532302857 (for 500 epochs)
Epoch  482  time  0.15791106224060059
Average time per epoch 0.20311047744750976 (for 500 epochs)
Epoch  483  time  0.1493685245513916
Average time per epoch 0.20340921449661256 (for 500 epochs)
Epoch  484  time  0.14660859107971191
Average time per epoch 0.20370243167877197 (for 500 epochs)
Epoch  485  time  0.14623022079467773
Average time per epoch 0.20399489212036132 (for 500 epochs)
Epoch  486  time  0.14335083961486816
Average time per epoch 0.20428159379959107 (for 500 epochs)
Epoch  487  time  0.14341950416564941
Average time per epoch 0.20456843280792236 (for 500 epochs)
Epoch  488  time  0.14074945449829102
Average time per epoch 0.20484993171691895 (for 500 epochs)
Epoch  489  time  0.15821003913879395
Average time per epoch 0.20516635179519654 (for 500 epochs)
Epoch  490  time  0.15429162979125977
Epoch  490  loss  0.18678818483815615 correct 50
Average time per epoch 0.20547493505477904 (for 500 epochs)
Epoch  491  time  0.14841365814208984
Average time per epoch 0.20577176237106323 (for 500 epochs)
Epoch  492  time  0.1446857452392578
Average time per epoch 0.20606113386154173 (for 500 epochs)
Epoch  493  time  0.1529710292816162
Average time per epoch 0.20636707592010498 (for 500 epochs)
Epoch  494  time  0.1394057273864746
Average time per epoch 0.20664588737487793 (for 500 epochs)
Epoch  495  time  0.16402578353881836
Average time per epoch 0.20697393894195557 (for 500 epochs)
Epoch  496  time  0.13948774337768555
Average time per epoch 0.20725291442871094 (for 500 epochs)
Epoch  497  time  0.15001368522644043
Average time per epoch 0.2075529417991638 (for 500 epochs)
Epoch  498  time  0.1494143009185791
Average time per epoch 0.20785177040100097 (for 500 epochs)
Epoch  499  time  0.1557767391204834
Average time per epoch 0.20816332387924194 (for 500 epochs)
```
# CPU Simple Large Dataset:
```
Epoch  0  time  21.367435932159424
Epoch  0  loss  10.416910102316477 correct 43
Average time per epoch 0.042734871864318846 (for 500 epochs)
Epoch  1  time  0.7866184711456299
Average time per epoch 0.04430810880661011 (for 500 epochs)
Epoch  2  time  0.7976207733154297
Average time per epoch 0.04590335035324097 (for 500 epochs)
Epoch  3  time  0.8769505023956299
Average time per epoch 0.04765725135803223 (for 500 epochs)
Epoch  4  time  0.7856001853942871
Average time per epoch 0.0492284517288208 (for 500 epochs)
Epoch  5  time  0.7858188152313232
Average time per epoch 0.05080008935928345 (for 500 epochs)
Epoch  6  time  0.7965178489685059
Average time per epoch 0.052393125057220456 (for 500 epochs)
Epoch  7  time  0.7910785675048828
Average time per epoch 0.053975282192230224 (for 500 epochs)
Epoch  8  time  0.7917568683624268
Average time per epoch 0.05555879592895508 (for 500 epochs)
Epoch  9  time  0.7931098937988281
Average time per epoch 0.057145015716552734 (for 500 epochs)
Epoch  10  time  0.7812955379486084
Epoch  10  loss  1.3662865126329617 correct 47
Average time per epoch 0.05870760679244995 (for 500 epochs)
Epoch  11  time  0.8396861553192139
Average time per epoch 0.06038697910308838 (for 500 epochs)
Epoch  12  time  0.7990908622741699
Average time per epoch 0.06198516082763672 (for 500 epochs)
Epoch  13  time  0.7869236469268799
Average time per epoch 0.06355900812149048 (for 500 epochs)
Epoch  14  time  0.7933290004730225
Average time per epoch 0.06514566612243652 (for 500 epochs)
Epoch  15  time  0.7749185562133789
Average time per epoch 0.06669550323486328 (for 500 epochs)
Epoch  16  time  0.8039402961730957
Average time per epoch 0.06830338382720948 (for 500 epochs)
Epoch  17  time  0.8146297931671143
Average time per epoch 0.0699326434135437 (for 500 epochs)
Epoch  18  time  0.794896125793457
Average time per epoch 0.07152243566513061 (for 500 epochs)
Epoch  19  time  0.7804524898529053
Average time per epoch 0.07308334064483643 (for 500 epochs)
Epoch  20  time  0.8319697380065918
Epoch  20  loss  1.7008260764104386 correct 49
Average time per epoch 0.07474728012084961 (for 500 epochs)
Epoch  21  time  0.8032307624816895
Average time per epoch 0.07635374164581299 (for 500 epochs)
Epoch  22  time  0.81467604637146
Average time per epoch 0.07798309373855591 (for 500 epochs)
Epoch  23  time  0.7905080318450928
Average time per epoch 0.0795641098022461 (for 500 epochs)
Epoch  24  time  0.8010869026184082
Average time per epoch 0.08116628360748292 (for 500 epochs)
Epoch  25  time  0.8000619411468506
Average time per epoch 0.08276640748977661 (for 500 epochs)
Epoch  26  time  0.7975814342498779
Average time per epoch 0.08436157035827636 (for 500 epochs)
Epoch  27  time  0.7933368682861328
Average time per epoch 0.08594824409484864 (for 500 epochs)
Epoch  28  time  0.8207578659057617
Average time per epoch 0.08758975982666016 (for 500 epochs)
Epoch  29  time  0.7981524467468262
Average time per epoch 0.08918606472015381 (for 500 epochs)
Epoch  30  time  0.7922449111938477
Epoch  30  loss  2.7750656372295937 correct 45
Average time per epoch 0.0907705545425415 (for 500 epochs)
Epoch  31  time  0.8124871253967285
Average time per epoch 0.09239552879333496 (for 500 epochs)
Epoch  32  time  0.7853212356567383
Average time per epoch 0.09396617126464844 (for 500 epochs)
Epoch  33  time  0.7911660671234131
Average time per epoch 0.09554850339889526 (for 500 epochs)
Epoch  34  time  0.8018252849578857
Average time per epoch 0.09715215396881104 (for 500 epochs)
Epoch  35  time  0.8001630306243896
Average time per epoch 0.09875248003005982 (for 500 epochs)
Epoch  36  time  0.8178954124450684
Average time per epoch 0.10038827085494995 (for 500 epochs)
Epoch  37  time  0.8392560482025146
Average time per epoch 0.10206678295135498 (for 500 epochs)
Epoch  38  time  0.8191688060760498
Average time per epoch 0.10370512056350709 (for 500 epochs)
Epoch  39  time  0.7979636192321777
Average time per epoch 0.10530104780197144 (for 500 epochs)
Epoch  40  time  0.8041415214538574
Epoch  40  loss  1.573826801194486 correct 49
Average time per epoch 0.10690933084487915 (for 500 epochs)
Epoch  41  time  0.7964658737182617
Average time per epoch 0.10850226259231567 (for 500 epochs)
Epoch  42  time  0.7998008728027344
Average time per epoch 0.11010186433792114 (for 500 epochs)
Epoch  43  time  0.7801339626312256
Average time per epoch 0.1116621322631836 (for 500 epochs)
Epoch  44  time  0.8034403324127197
Average time per epoch 0.11326901292800903 (for 500 epochs)
Epoch  45  time  0.7922830581665039
Average time per epoch 0.11485357904434204 (for 500 epochs)
Epoch  46  time  0.7954707145690918
Average time per epoch 0.11644452047348022 (for 500 epochs)
Epoch  47  time  0.7810392379760742
Average time per epoch 0.11800659894943237 (for 500 epochs)
Epoch  48  time  0.7807502746582031
Average time per epoch 0.11956809949874878 (for 500 epochs)
Epoch  49  time  0.7944579124450684
Average time per epoch 0.12115701532363891 (for 500 epochs)
Epoch  50  time  0.8203351497650146
Epoch  50  loss  0.2506364741751463 correct 49
Average time per epoch 0.12279768562316895 (for 500 epochs)
Epoch  51  time  0.8203229904174805
Average time per epoch 0.1244383316040039 (for 500 epochs)
Epoch  52  time  0.7782728672027588
Average time per epoch 0.12599487733840942 (for 500 epochs)
Epoch  53  time  0.8083484172821045
Average time per epoch 0.12761157417297364 (for 500 epochs)
Epoch  54  time  0.811518669128418
Average time per epoch 0.12923461151123047 (for 500 epochs)
Epoch  55  time  0.804072380065918
Average time per epoch 0.1308427562713623 (for 500 epochs)
Epoch  56  time  0.7957797050476074
Average time per epoch 0.13243431568145753 (for 500 epochs)
Epoch  57  time  0.7767078876495361
Average time per epoch 0.1339877314567566 (for 500 epochs)
Epoch  58  time  0.8122000694274902
Average time per epoch 0.13561213159561158 (for 500 epochs)
Epoch  59  time  0.797170877456665
Average time per epoch 0.1372064733505249 (for 500 epochs)
Epoch  60  time  0.7850246429443359
Epoch  60  loss  0.13696141613626595 correct 49
Average time per epoch 0.13877652263641357 (for 500 epochs)
Epoch  61  time  0.7938873767852783
Average time per epoch 0.14036429738998413 (for 500 epochs)
Epoch  62  time  0.7992267608642578
Average time per epoch 0.14196275091171265 (for 500 epochs)
Epoch  63  time  0.8100523948669434
Average time per epoch 0.14358285570144652 (for 500 epochs)
Epoch  64  time  0.7892184257507324
Average time per epoch 0.145161292552948 (for 500 epochs)
Epoch  65  time  0.7861204147338867
Average time per epoch 0.14673353338241577 (for 500 epochs)
Epoch  66  time  0.7637522220611572
Average time per epoch 0.14826103782653807 (for 500 epochs)
Epoch  67  time  0.7732741832733154
Average time per epoch 0.1498075861930847 (for 500 epochs)
Epoch  68  time  0.7872986793518066
Average time per epoch 0.15138218355178834 (for 500 epochs)
Epoch  69  time  0.7983639240264893
Average time per epoch 0.15297891139984132 (for 500 epochs)
Epoch  70  time  0.8306190967559814
Epoch  70  loss  2.4834752023341613 correct 49
Average time per epoch 0.15464014959335326 (for 500 epochs)
Epoch  71  time  3.8594982624053955
Average time per epoch 0.16235914611816407 (for 500 epochs)
Epoch  72  time  3.392089366912842
Average time per epoch 0.16914332485198974 (for 500 epochs)
Epoch  73  time  3.542707681655884
Average time per epoch 0.17622874021530152 (for 500 epochs)
Epoch  74  time  1.478968858718872
Average time per epoch 0.17918667793273926 (for 500 epochs)
Epoch  75  time  0.7883036136627197
Average time per epoch 0.1807632851600647 (for 500 epochs)
Epoch  76  time  0.7874877452850342
Average time per epoch 0.18233826065063477 (for 500 epochs)
Epoch  77  time  0.7947885990142822
Average time per epoch 0.18392783784866332 (for 500 epochs)
Epoch  78  time  0.8049161434173584
Average time per epoch 0.18553767013549805 (for 500 epochs)
Epoch  79  time  0.8214597702026367
Average time per epoch 0.1871805896759033 (for 500 epochs)
Epoch  80  time  0.8027865886688232
Epoch  80  loss  0.3757235485342516 correct 50
Average time per epoch 0.18878616285324096 (for 500 epochs)
Epoch  81  time  0.8184306621551514
Average time per epoch 0.19042302417755128 (for 500 epochs)
Epoch  82  time  0.7981503009796143
Average time per epoch 0.1920193247795105 (for 500 epochs)
Epoch  83  time  0.774172306060791
Average time per epoch 0.19356766939163209 (for 500 epochs)
Epoch  84  time  0.8286628723144531
Average time per epoch 0.195224995136261 (for 500 epochs)
Epoch  85  time  0.8342630863189697
Average time per epoch 0.19689352130889892 (for 500 epochs)
Epoch  86  time  0.7977774143218994
Average time per epoch 0.1984890761375427 (for 500 epochs)
Epoch  87  time  0.7968010902404785
Average time per epoch 0.20008267831802368 (for 500 epochs)
Epoch  88  time  0.7652091979980469
Average time per epoch 0.20161309671401978 (for 500 epochs)
Epoch  89  time  0.807593822479248
Average time per epoch 0.20322828435897827 (for 500 epochs)
Epoch  90  time  0.8011512756347656
Epoch  90  loss  1.3979321328201029 correct 49
Average time per epoch 0.2048305869102478 (for 500 epochs)
Epoch  91  time  0.7897264957427979
Average time per epoch 0.2064100399017334 (for 500 epochs)
Epoch  92  time  0.8423688411712646
Average time per epoch 0.20809477758407594 (for 500 epochs)
Epoch  93  time  0.7799327373504639
Average time per epoch 0.20965464305877685 (for 500 epochs)
Epoch  94  time  0.802314043045044
Average time per epoch 0.21125927114486695 (for 500 epochs)
Epoch  95  time  0.8065869808197021
Average time per epoch 0.21287244510650635 (for 500 epochs)
Epoch  96  time  0.7861909866333008
Average time per epoch 0.21444482707977294 (for 500 epochs)
Epoch  97  time  0.7669141292572021
Average time per epoch 0.21597865533828736 (for 500 epochs)
Epoch  98  time  0.8005073070526123
Average time per epoch 0.21757966995239258 (for 500 epochs)
Epoch  99  time  0.8176171779632568
Average time per epoch 0.21921490430831908 (for 500 epochs)
Epoch  100  time  0.800201416015625
Epoch  100  loss  1.314592587541807 correct 49
Average time per epoch 0.22081530714035033 (for 500 epochs)
Epoch  101  time  0.8021450042724609
Average time per epoch 0.22241959714889525 (for 500 epochs)
Epoch  102  time  0.7761516571044922
Average time per epoch 0.22397190046310425 (for 500 epochs)
Epoch  103  time  0.7843625545501709
Average time per epoch 0.22554062557220458 (for 500 epochs)
Epoch  104  time  0.8066229820251465
Average time per epoch 0.22715387153625488 (for 500 epochs)
Epoch  105  time  0.7937171459197998
Average time per epoch 0.2287413058280945 (for 500 epochs)
Epoch  106  time  0.7800197601318359
Average time per epoch 0.23030134534835817 (for 500 epochs)
Epoch  107  time  0.7937328815460205
Average time per epoch 0.23188881111145018 (for 500 epochs)
Epoch  108  time  0.7847850322723389
Average time per epoch 0.23345838117599488 (for 500 epochs)
Epoch  109  time  0.7863826751708984
Average time per epoch 0.23503114652633667 (for 500 epochs)
Epoch  110  time  0.8014843463897705
Epoch  110  loss  0.13082970830005286 correct 49
Average time per epoch 0.2366341152191162 (for 500 epochs)
Epoch  111  time  0.7572448253631592
Average time per epoch 0.23814860486984252 (for 500 epochs)
Epoch  112  time  0.7829370498657227
Average time per epoch 0.23971447896957399 (for 500 epochs)
Epoch  113  time  0.7776260375976562
Average time per epoch 0.2412697310447693 (for 500 epochs)
Epoch  114  time  0.780400276184082
Average time per epoch 0.24283053159713744 (for 500 epochs)
Epoch  115  time  0.7708609104156494
Average time per epoch 0.24437225341796875 (for 500 epochs)
Epoch  116  time  0.7851963043212891
Average time per epoch 0.24594264602661134 (for 500 epochs)
Epoch  117  time  0.8071649074554443
Average time per epoch 0.24755697584152223 (for 500 epochs)
Epoch  118  time  0.7986798286437988
Average time per epoch 0.2491543354988098 (for 500 epochs)
Epoch  119  time  0.7831892967224121
Average time per epoch 0.25072071409225466 (for 500 epochs)
Epoch  120  time  0.7682151794433594
Epoch  120  loss  0.01100218067675053 correct 50
Average time per epoch 0.25225714445114134 (for 500 epochs)
Epoch  121  time  0.7794582843780518
Average time per epoch 0.25381606101989745 (for 500 epochs)
Epoch  122  time  0.773770809173584
Average time per epoch 0.25536360263824465 (for 500 epochs)
Epoch  123  time  0.8200294971466064
Average time per epoch 0.25700366163253785 (for 500 epochs)
Epoch  124  time  0.7665562629699707
Average time per epoch 0.25853677415847776 (for 500 epochs)
Epoch  125  time  0.8362846374511719
Average time per epoch 0.26020934343338015 (for 500 epochs)
Epoch  126  time  0.8067402839660645
Average time per epoch 0.26182282400131224 (for 500 epochs)
Epoch  127  time  0.7812488079071045
Average time per epoch 0.26338532161712647 (for 500 epochs)
Epoch  128  time  0.7723033428192139
Average time per epoch 0.2649299283027649 (for 500 epochs)
Epoch  129  time  0.7746577262878418
Average time per epoch 0.2664792437553406 (for 500 epochs)
Epoch  130  time  0.8116810321807861
Epoch  130  loss  1.0261548711865163 correct 49
Average time per epoch 0.26810260581970213 (for 500 epochs)
Epoch  131  time  0.7730245590209961
Average time per epoch 0.26964865493774415 (for 500 epochs)
Epoch  132  time  0.7854602336883545
Average time per epoch 0.27121957540512087 (for 500 epochs)
Epoch  133  time  0.7707664966583252
Average time per epoch 0.2727611083984375 (for 500 epochs)
Epoch  134  time  0.7805767059326172
Average time per epoch 0.2743222618103027 (for 500 epochs)
Epoch  135  time  0.7852146625518799
Average time per epoch 0.2758926911354065 (for 500 epochs)
Epoch  136  time  0.7912425994873047
Average time per epoch 0.2774751763343811 (for 500 epochs)
Epoch  137  time  0.7820587158203125
Average time per epoch 0.2790392937660217 (for 500 epochs)
Epoch  138  time  0.7809174060821533
Average time per epoch 0.28060112857818603 (for 500 epochs)
Epoch  139  time  0.7980139255523682
Average time per epoch 0.2821971564292908 (for 500 epochs)
Epoch  140  time  0.7925059795379639
Epoch  140  loss  1.407284950869562 correct 49
Average time per epoch 0.2837821683883667 (for 500 epochs)
Epoch  141  time  0.7992496490478516
Average time per epoch 0.2853806676864624 (for 500 epochs)
Epoch  142  time  0.7836532592773438
Average time per epoch 0.2869479742050171 (for 500 epochs)
Epoch  143  time  0.8171300888061523
Average time per epoch 0.2885822343826294 (for 500 epochs)
Epoch  144  time  0.8002109527587891
Average time per epoch 0.290182656288147 (for 500 epochs)
Epoch  145  time  0.8056628704071045
Average time per epoch 0.2917939820289612 (for 500 epochs)
Epoch  146  time  0.7980434894561768
Average time per epoch 0.2933900690078735 (for 500 epochs)
Epoch  147  time  2.954589605331421
Average time per epoch 0.2992992482185364 (for 500 epochs)
Epoch  148  time  3.9417824745178223
Average time per epoch 0.307182813167572 (for 500 epochs)
Epoch  149  time  5.048183441162109
Average time per epoch 0.31727918004989625 (for 500 epochs)
Epoch  150  time  0.9141323566436768
Epoch  150  loss  0.5836775798485357 correct 49
Average time per epoch 0.3191074447631836 (for 500 epochs)
Epoch  151  time  0.789006233215332
Average time per epoch 0.3206854572296143 (for 500 epochs)
Epoch  152  time  0.785327672958374
Average time per epoch 0.322256112575531 (for 500 epochs)
Epoch  153  time  0.766193151473999
Average time per epoch 0.323788498878479 (for 500 epochs)
Epoch  154  time  0.8087570667266846
Average time per epoch 0.3254060130119324 (for 500 epochs)
Epoch  155  time  0.788088321685791
Average time per epoch 0.32698218965530396 (for 500 epochs)
Epoch  156  time  0.7765603065490723
Average time per epoch 0.3285353102684021 (for 500 epochs)
Epoch  157  time  0.7661349773406982
Average time per epoch 0.3300675802230835 (for 500 epochs)
Epoch  158  time  0.8217694759368896
Average time per epoch 0.33171111917495727 (for 500 epochs)
Epoch  159  time  0.7724368572235107
Average time per epoch 0.3332559928894043 (for 500 epochs)
Epoch  160  time  0.7782974243164062
Epoch  160  loss  1.1285232512944925 correct 49
Average time per epoch 0.3348125877380371 (for 500 epochs)
Epoch  161  time  0.7926943302154541
Average time per epoch 0.336397976398468 (for 500 epochs)
Epoch  162  time  0.775686502456665
Average time per epoch 0.33794934940338134 (for 500 epochs)
Epoch  163  time  0.7809257507324219
Average time per epoch 0.3395112009048462 (for 500 epochs)
Epoch  164  time  0.7864720821380615
Average time per epoch 0.3410841450691223 (for 500 epochs)
Epoch  165  time  0.7811427116394043
Average time per epoch 0.3426464304924011 (for 500 epochs)
Epoch  166  time  0.7669358253479004
Average time per epoch 0.3441803021430969 (for 500 epochs)
Epoch  167  time  0.8076751232147217
Average time per epoch 0.34579565238952636 (for 500 epochs)
Epoch  168  time  0.7805583477020264
Average time per epoch 0.3473567690849304 (for 500 epochs)
Epoch  169  time  0.7859830856323242
Average time per epoch 0.34892873525619506 (for 500 epochs)
Epoch  170  time  0.7616548538208008
Epoch  170  loss  1.1166064662749604 correct 49
Average time per epoch 0.3504520449638367 (for 500 epochs)
Epoch  171  time  0.768700122833252
Average time per epoch 0.35198944520950315 (for 500 epochs)
Epoch  172  time  0.8044857978820801
Average time per epoch 0.3535984168052673 (for 500 epochs)
Epoch  173  time  0.7793080806732178
Average time per epoch 0.3551570329666138 (for 500 epochs)
Epoch  174  time  0.7775077819824219
Average time per epoch 0.35671204853057864 (for 500 epochs)
Epoch  175  time  0.7912421226501465
Average time per epoch 0.3582945327758789 (for 500 epochs)
Epoch  176  time  0.7865235805511475
Average time per epoch 0.3598675799369812 (for 500 epochs)
Epoch  177  time  1.6505227088928223
Average time per epoch 0.36316862535476685 (for 500 epochs)
Epoch  178  time  3.303551435470581
Average time per epoch 0.36977572822570803 (for 500 epochs)
Epoch  179  time  3.4081380367279053
Average time per epoch 0.37659200429916384 (for 500 epochs)
Epoch  180  time  4.274151802062988
Epoch  180  loss  0.16121109123060984 correct 49
Average time per epoch 0.3851403079032898 (for 500 epochs)
Epoch  181  time  0.7833833694458008
Average time per epoch 0.3867070746421814 (for 500 epochs)
Epoch  182  time  0.7634522914886475
Average time per epoch 0.3882339792251587 (for 500 epochs)
Epoch  183  time  0.7756397724151611
Average time per epoch 0.38978525876998904 (for 500 epochs)
Epoch  184  time  0.7802493572235107
Average time per epoch 0.391345757484436 (for 500 epochs)
Epoch  185  time  0.7904672622680664
Average time per epoch 0.3929266920089722 (for 500 epochs)
Epoch  186  time  0.7731378078460693
Average time per epoch 0.39447296762466433 (for 500 epochs)
Epoch  187  time  0.7745110988616943
Average time per epoch 0.3960219898223877 (for 500 epochs)
Epoch  188  time  0.7655153274536133
Average time per epoch 0.3975530204772949 (for 500 epochs)
Epoch  189  time  0.7886145114898682
Average time per epoch 0.39913024950027465 (for 500 epochs)
Epoch  190  time  0.7755105495452881
Epoch  190  loss  0.18435095057034553 correct 49
Average time per epoch 0.40068127059936526 (for 500 epochs)
Epoch  191  time  0.7955985069274902
Average time per epoch 0.4022724676132202 (for 500 epochs)
Epoch  192  time  0.7675144672393799
Average time per epoch 0.403807496547699 (for 500 epochs)
Epoch  193  time  0.7867133617401123
Average time per epoch 0.4053809232711792 (for 500 epochs)
Epoch  194  time  0.788590669631958
Average time per epoch 0.4069581046104431 (for 500 epochs)
Epoch  195  time  0.7876152992248535
Average time per epoch 0.40853333520889284 (for 500 epochs)
Epoch  196  time  0.7879807949066162
Average time per epoch 0.41010929679870606 (for 500 epochs)
Epoch  197  time  0.7602179050445557
Average time per epoch 0.41162973260879515 (for 500 epochs)
Epoch  198  time  0.7802977561950684
Average time per epoch 0.4131903281211853 (for 500 epochs)
Epoch  199  time  0.7839610576629639
Average time per epoch 0.41475825023651125 (for 500 epochs)
Epoch  200  time  0.7803893089294434
Epoch  200  loss  0.07918861823226728 correct 49
Average time per epoch 0.4163190288543701 (for 500 epochs)
Epoch  201  time  0.7689006328582764
Average time per epoch 0.4178568301200867 (for 500 epochs)
Epoch  202  time  0.7779080867767334
Average time per epoch 0.4194126462936401 (for 500 epochs)
Epoch  203  time  0.8135690689086914
Average time per epoch 0.4210397844314575 (for 500 epochs)
Epoch  204  time  0.8113956451416016
Average time per epoch 0.4226625757217407 (for 500 epochs)
Epoch  205  time  0.7628405094146729
Average time per epoch 0.4241882567405701 (for 500 epochs)
Epoch  206  time  0.7749366760253906
Average time per epoch 0.42573813009262085 (for 500 epochs)
Epoch  207  time  0.8039634227752686
Average time per epoch 0.4273460569381714 (for 500 epochs)
Epoch  208  time  0.7920398712158203
Average time per epoch 0.42893013668060304 (for 500 epochs)
Epoch  209  time  0.7871816158294678
Average time per epoch 0.430504499912262 (for 500 epochs)
Epoch  210  time  0.7631340026855469
Epoch  210  loss  0.05923841991624237 correct 49
Average time per epoch 0.43203076791763306 (for 500 epochs)
Epoch  211  time  0.8061223030090332
Average time per epoch 0.4336430125236511 (for 500 epochs)
Epoch  212  time  0.7860350608825684
Average time per epoch 0.43521508264541625 (for 500 epochs)
Epoch  213  time  0.8059306144714355
Average time per epoch 0.4368269438743591 (for 500 epochs)
Epoch  214  time  0.8470516204833984
Average time per epoch 0.4385210471153259 (for 500 epochs)
Epoch  215  time  2.078238010406494
Average time per epoch 0.4426775231361389 (for 500 epochs)
Epoch  216  time  5.810066223144531
Average time per epoch 0.45429765558242796 (for 500 epochs)
Epoch  217  time  4.424216032028198
Average time per epoch 0.46314608764648435 (for 500 epochs)
Epoch  218  time  0.7926814556121826
Average time per epoch 0.46473145055770876 (for 500 epochs)
Epoch  219  time  0.7890784740447998
Average time per epoch 0.46630960750579836 (for 500 epochs)
Epoch  220  time  0.7715060710906982
Epoch  220  loss  0.018109628196621864 correct 50
Average time per epoch 0.46785261964797975 (for 500 epochs)
Epoch  221  time  0.7749686241149902
Average time per epoch 0.46940255689620974 (for 500 epochs)
Epoch  222  time  0.7860367298126221
Average time per epoch 0.470974630355835 (for 500 epochs)
Epoch  223  time  0.8002660274505615
Average time per epoch 0.4725751624107361 (for 500 epochs)
Epoch  224  time  0.789013147354126
Average time per epoch 0.4741531887054443 (for 500 epochs)
Epoch  225  time  0.8468048572540283
Average time per epoch 0.47584679841995237 (for 500 epochs)
Epoch  226  time  0.7781815528869629
Average time per epoch 0.4774031615257263 (for 500 epochs)
Epoch  227  time  0.7816517353057861
Average time per epoch 0.4789664649963379 (for 500 epochs)
Epoch  228  time  0.7951126098632812
Average time per epoch 0.48055669021606445 (for 500 epochs)
Epoch  229  time  0.7839145660400391
Average time per epoch 0.48212451934814454 (for 500 epochs)
Epoch  230  time  0.7949244976043701
Epoch  230  loss  1.2843058400115408 correct 49
Average time per epoch 0.4837143683433533 (for 500 epochs)
Epoch  231  time  0.8497927188873291
Average time per epoch 0.48541395378112795 (for 500 epochs)
Epoch  232  time  3.774174213409424
Average time per epoch 0.4929623022079468 (for 500 epochs)
Epoch  233  time  3.3248534202575684
Average time per epoch 0.4996120090484619 (for 500 epochs)
Epoch  234  time  4.522400140762329
Average time per epoch 0.5086568093299866 (for 500 epochs)
Epoch  235  time  0.7716500759124756
Average time per epoch 0.5102001094818115 (for 500 epochs)
Epoch  236  time  0.8017387390136719
Average time per epoch 0.5118035869598389 (for 500 epochs)
Epoch  237  time  0.8054854869842529
Average time per epoch 0.5134145579338074 (for 500 epochs)
Epoch  238  time  0.779526948928833
Average time per epoch 0.514973611831665 (for 500 epochs)
Epoch  239  time  0.7821967601776123
Average time per epoch 0.5165380053520202 (for 500 epochs)
Epoch  240  time  0.7922878265380859
Epoch  240  loss  1.4605997144840723 correct 49
Average time per epoch 0.5181225810050964 (for 500 epochs)
Epoch  241  time  0.780963659286499
Average time per epoch 0.5196845083236694 (for 500 epochs)
Epoch  242  time  0.7815077304840088
Average time per epoch 0.5212475237846375 (for 500 epochs)
Epoch  243  time  0.7959015369415283
Average time per epoch 0.5228393268585205 (for 500 epochs)
Epoch  244  time  0.7782495021820068
Average time per epoch 0.5243958258628845 (for 500 epochs)
Epoch  245  time  0.7775304317474365
Average time per epoch 0.5259508867263794 (for 500 epochs)
Epoch  246  time  0.7982978820800781
Average time per epoch 0.5275474824905395 (for 500 epochs)
Epoch  247  time  0.7909443378448486
Average time per epoch 0.5291293711662293 (for 500 epochs)
Epoch  248  time  0.7892467975616455
Average time per epoch 0.5307078647613526 (for 500 epochs)
Epoch  249  time  0.7829587459564209
Average time per epoch 0.5322737822532654 (for 500 epochs)
Epoch  250  time  0.7925126552581787
Epoch  250  loss  0.13117273580411526 correct 49
Average time per epoch 0.5338588075637818 (for 500 epochs)
Epoch  251  time  0.7760589122772217
Average time per epoch 0.5354109253883362 (for 500 epochs)
Epoch  252  time  0.7863805294036865
Average time per epoch 0.5369836864471436 (for 500 epochs)
Epoch  253  time  0.7937517166137695
Average time per epoch 0.5385711898803711 (for 500 epochs)
Epoch  254  time  0.779832124710083
Average time per epoch 0.5401308541297912 (for 500 epochs)
Epoch  255  time  0.7793657779693604
Average time per epoch 0.54168958568573 (for 500 epochs)
Epoch  256  time  0.7791314125061035
Average time per epoch 0.5432478485107421 (for 500 epochs)
Epoch  257  time  0.7688724994659424
Average time per epoch 0.544785593509674 (for 500 epochs)
Epoch  258  time  0.8190834522247314
Average time per epoch 0.5464237604141235 (for 500 epochs)
Epoch  259  time  0.787977933883667
Average time per epoch 0.5479997162818908 (for 500 epochs)
Epoch  260  time  0.7854971885681152
Epoch  260  loss  1.3140488902486827 correct 49
Average time per epoch 0.5495707106590271 (for 500 epochs)
Epoch  261  time  0.7740728855133057
Average time per epoch 0.5511188564300538 (for 500 epochs)
Epoch  262  time  0.78232741355896
Average time per epoch 0.5526835112571716 (for 500 epochs)
Epoch  263  time  0.795487642288208
Average time per epoch 0.554274486541748 (for 500 epochs)
Epoch  264  time  0.7956311702728271
Average time per epoch 0.5558657488822937 (for 500 epochs)
Epoch  265  time  0.78397536277771
Average time per epoch 0.5574336996078492 (for 500 epochs)
Epoch  266  time  0.864938497543335
Average time per epoch 0.5591635766029358 (for 500 epochs)
Epoch  267  time  0.7814614772796631
Average time per epoch 0.5607264995574951 (for 500 epochs)
Epoch  268  time  0.7867281436920166
Average time per epoch 0.5622999558448791 (for 500 epochs)
Epoch  269  time  0.7857496738433838
Average time per epoch 0.5638714551925659 (for 500 epochs)
Epoch  270  time  0.7828855514526367
Epoch  270  loss  0.201883819339225 correct 49
Average time per epoch 0.5654372262954712 (for 500 epochs)
Epoch  271  time  0.7622232437133789
Average time per epoch 0.566961672782898 (for 500 epochs)
Epoch  272  time  0.7863442897796631
Average time per epoch 0.5685343613624573 (for 500 epochs)
Epoch  273  time  0.8154969215393066
Average time per epoch 0.5701653552055359 (for 500 epochs)
Epoch  274  time  0.7867450714111328
Average time per epoch 0.5717388453483582 (for 500 epochs)
Epoch  275  time  0.7697274684906006
Average time per epoch 0.5732783002853393 (for 500 epochs)
Epoch  276  time  0.7873802185058594
Average time per epoch 0.5748530607223511 (for 500 epochs)
Epoch  277  time  0.7934770584106445
Average time per epoch 0.5764400148391724 (for 500 epochs)
Epoch  278  time  0.8031392097473145
Average time per epoch 0.578046293258667 (for 500 epochs)
Epoch  279  time  0.8029301166534424
Average time per epoch 0.5796521534919739 (for 500 epochs)
Epoch  280  time  0.7929902076721191
Epoch  280  loss  1.1410945231296064 correct 49
Average time per epoch 0.5812381339073182 (for 500 epochs)
Epoch  281  time  0.8346829414367676
Average time per epoch 0.5829074997901916 (for 500 epochs)
Epoch  282  time  2.452086925506592
Average time per epoch 0.5878116736412048 (for 500 epochs)
Epoch  283  time  4.808240175247192
Average time per epoch 0.5974281539916992 (for 500 epochs)
Epoch  284  time  3.88737416267395
Average time per epoch 0.6052029023170471 (for 500 epochs)
Epoch  285  time  1.6008696556091309
Average time per epoch 0.6084046416282654 (for 500 epochs)
Epoch  286  time  0.781095027923584
Average time per epoch 0.6099668316841126 (for 500 epochs)
Epoch  287  time  0.7891719341278076
Average time per epoch 0.6115451755523682 (for 500 epochs)
Epoch  288  time  0.7897214889526367
Average time per epoch 0.6131246185302734 (for 500 epochs)
Epoch  289  time  0.792809009552002
Average time per epoch 0.6147102365493774 (for 500 epochs)
Epoch  290  time  0.7699835300445557
Epoch  290  loss  0.8887986092996778 correct 49
Average time per epoch 0.6162502036094666 (for 500 epochs)
Epoch  291  time  0.8008809089660645
Average time per epoch 0.6178519654273987 (for 500 epochs)
Epoch  292  time  0.7971210479736328
Average time per epoch 0.6194462075233459 (for 500 epochs)
Epoch  293  time  0.8161313533782959
Average time per epoch 0.6210784702301025 (for 500 epochs)
Epoch  294  time  0.787834644317627
Average time per epoch 0.6226541395187378 (for 500 epochs)
Epoch  295  time  0.8044426441192627
Average time per epoch 0.6242630248069763 (for 500 epochs)
Epoch  296  time  0.7840771675109863
Average time per epoch 0.6258311791419983 (for 500 epochs)
Epoch  297  time  0.782367467880249
Average time per epoch 0.6273959140777587 (for 500 epochs)
Epoch  298  time  0.7723150253295898
Average time per epoch 0.6289405441284179 (for 500 epochs)
Epoch  299  time  0.7782177925109863
Average time per epoch 0.63049697971344 (for 500 epochs)
Epoch  300  time  0.7852697372436523
Epoch  300  loss  1.4901467318237782 correct 49
Average time per epoch 0.6320675191879273 (for 500 epochs)
Epoch  301  time  0.7887752056121826
Average time per epoch 0.6336450695991516 (for 500 epochs)
Epoch  302  time  0.7796473503112793
Average time per epoch 0.6352043642997741 (for 500 epochs)
Epoch  303  time  0.7741096019744873
Average time per epoch 0.6367525835037231 (for 500 epochs)
Epoch  304  time  0.8005654811859131
Average time per epoch 0.638353714466095 (for 500 epochs)
Epoch  305  time  0.7782046794891357
Average time per epoch 0.6399101238250733 (for 500 epochs)
Epoch  306  time  0.8036599159240723
Average time per epoch 0.6415174436569214 (for 500 epochs)
Epoch  307  time  0.7850334644317627
Average time per epoch 0.643087510585785 (for 500 epochs)
Epoch  308  time  0.7627761363983154
Average time per epoch 0.6446130628585816 (for 500 epochs)
Epoch  309  time  0.7855038642883301
Average time per epoch 0.6461840705871582 (for 500 epochs)
Epoch  310  time  0.8126928806304932
Epoch  310  loss  0.8469779459202754 correct 49
Average time per epoch 0.6478094563484192 (for 500 epochs)
Epoch  311  time  0.793562650680542
Average time per epoch 0.6493965816497803 (for 500 epochs)
Epoch  312  time  0.7727804183959961
Average time per epoch 0.6509421424865722 (for 500 epochs)
Epoch  313  time  0.762561559677124
Average time per epoch 0.6524672656059265 (for 500 epochs)
Epoch  314  time  0.7707066535949707
Average time per epoch 0.6540086789131164 (for 500 epochs)
Epoch  315  time  0.7780401706695557
Average time per epoch 0.6555647592544556 (for 500 epochs)
Epoch  316  time  0.7724905014038086
Average time per epoch 0.6571097402572632 (for 500 epochs)
Epoch  317  time  0.8008356094360352
Average time per epoch 0.6587114114761352 (for 500 epochs)
Epoch  318  time  0.7872200012207031
Average time per epoch 0.6602858514785767 (for 500 epochs)
Epoch  319  time  0.7857613563537598
Average time per epoch 0.6618573741912842 (for 500 epochs)
Epoch  320  time  0.8074123859405518
Epoch  320  loss  0.9984889685540922 correct 49
Average time per epoch 0.6634721989631653 (for 500 epochs)
Epoch  321  time  0.7660808563232422
Average time per epoch 0.6650043606758118 (for 500 epochs)
Epoch  322  time  0.8173117637634277
Average time per epoch 0.6666389842033387 (for 500 epochs)
Epoch  323  time  0.7975344657897949
Average time per epoch 0.6682340531349182 (for 500 epochs)
Epoch  324  time  0.8019459247589111
Average time per epoch 0.669837944984436 (for 500 epochs)
Epoch  325  time  0.8054213523864746
Average time per epoch 0.671448787689209 (for 500 epochs)
Epoch  326  time  0.7884457111358643
Average time per epoch 0.6730256791114807 (for 500 epochs)
Epoch  327  time  0.7935001850128174
Average time per epoch 0.6746126794815064 (for 500 epochs)
Epoch  328  time  0.7898991107940674
Average time per epoch 0.6761924777030944 (for 500 epochs)
Epoch  329  time  0.7981867790222168
Average time per epoch 0.677788851261139 (for 500 epochs)
Epoch  330  time  0.7853336334228516
Epoch  330  loss  1.2726033385331004 correct 49
Average time per epoch 0.6793595185279846 (for 500 epochs)
Epoch  331  time  0.798980712890625
Average time per epoch 0.6809574799537659 (for 500 epochs)
Epoch  332  time  0.8601548671722412
Average time per epoch 0.6826777896881103 (for 500 epochs)
Epoch  333  time  0.8160464763641357
Average time per epoch 0.6843098826408386 (for 500 epochs)
Epoch  334  time  0.7887940406799316
Average time per epoch 0.6858874707221985 (for 500 epochs)
Epoch  335  time  0.7705638408660889
Average time per epoch 0.6874285984039307 (for 500 epochs)
Epoch  336  time  0.8060288429260254
Average time per epoch 0.6890406560897827 (for 500 epochs)
Epoch  337  time  0.7968590259552002
Average time per epoch 0.6906343741416932 (for 500 epochs)
Epoch  338  time  0.7768921852111816
Average time per epoch 0.6921881585121155 (for 500 epochs)
Epoch  339  time  0.7860062122344971
Average time per epoch 0.6937601709365845 (for 500 epochs)
Epoch  340  time  0.809455156326294
Epoch  340  loss  0.03690696337766276 correct 49
Average time per epoch 0.6953790812492371 (for 500 epochs)
Epoch  341  time  0.8005366325378418
Average time per epoch 0.6969801545143127 (for 500 epochs)
Epoch  342  time  0.7935833930969238
Average time per epoch 0.6985673213005066 (for 500 epochs)
Epoch  343  time  0.8092503547668457
Average time per epoch 0.7001858220100403 (for 500 epochs)
Epoch  344  time  0.780827522277832
Average time per epoch 0.701747477054596 (for 500 epochs)
Epoch  345  time  0.7931957244873047
Average time per epoch 0.7033338685035706 (for 500 epochs)
Epoch  346  time  0.8208954334259033
Average time per epoch 0.7049756593704224 (for 500 epochs)
Epoch  347  time  0.831005334854126
Average time per epoch 0.7066376700401306 (for 500 epochs)
Epoch  348  time  0.7840955257415771
Average time per epoch 0.7082058610916138 (for 500 epochs)
Epoch  349  time  0.7845685482025146
Average time per epoch 0.7097749981880188 (for 500 epochs)
Epoch  350  time  0.7847676277160645
Epoch  350  loss  0.056681050267796136 correct 49
Average time per epoch 0.7113445334434509 (for 500 epochs)
Epoch  351  time  0.7917675971984863
Average time per epoch 0.7129280686378479 (for 500 epochs)
Epoch  352  time  0.8009848594665527
Average time per epoch 0.714530038356781 (for 500 epochs)
Epoch  353  time  0.7766673564910889
Average time per epoch 0.7160833730697632 (for 500 epochs)
Epoch  354  time  0.7917065620422363
Average time per epoch 0.7176667861938476 (for 500 epochs)
Epoch  355  time  0.7885081768035889
Average time per epoch 0.7192438025474548 (for 500 epochs)
Epoch  356  time  0.80438232421875
Average time per epoch 0.7208525671958923 (for 500 epochs)
Epoch  357  time  0.8174839019775391
Average time per epoch 0.7224875349998474 (for 500 epochs)
Epoch  358  time  1.3561804294586182
Average time per epoch 0.7251998958587647 (for 500 epochs)
Epoch  359  time  3.740384340286255
Average time per epoch 0.7326806645393371 (for 500 epochs)
Epoch  360  time  3.12054705619812
Epoch  360  loss  1.421092405037518 correct 49
Average time per epoch 0.7389217586517334 (for 500 epochs)
Epoch  361  time  4.056515216827393
Average time per epoch 0.7470347890853882 (for 500 epochs)
Epoch  362  time  0.7623705863952637
Average time per epoch 0.7485595302581787 (for 500 epochs)
Epoch  363  time  0.7862353324890137
Average time per epoch 0.7501320009231567 (for 500 epochs)
Epoch  364  time  0.7917318344116211
Average time per epoch 0.75171546459198 (for 500 epochs)
Epoch  365  time  0.7871096134185791
Average time per epoch 0.7532896838188171 (for 500 epochs)
Epoch  366  time  0.778275728225708
Average time per epoch 0.7548462352752685 (for 500 epochs)
Epoch  367  time  0.789008617401123
Average time per epoch 0.7564242525100708 (for 500 epochs)
Epoch  368  time  0.776594877243042
Average time per epoch 0.7579774422645569 (for 500 epochs)
Epoch  369  time  0.7901525497436523
Average time per epoch 0.7595577473640442 (for 500 epochs)
Epoch  370  time  0.7959773540496826
Epoch  370  loss  0.002098889162837464 correct 50
Average time per epoch 0.7611497020721436 (for 500 epochs)
Epoch  371  time  0.7813475131988525
Average time per epoch 0.7627123970985412 (for 500 epochs)
Epoch  372  time  0.8163044452667236
Average time per epoch 0.7643450059890747 (for 500 epochs)
Epoch  373  time  0.8016071319580078
Average time per epoch 0.7659482202529907 (for 500 epochs)
Epoch  374  time  0.8031473159790039
Average time per epoch 0.7675545148849487 (for 500 epochs)
Epoch  375  time  0.8069164752960205
Average time per epoch 0.7691683478355408 (for 500 epochs)
Epoch  376  time  0.7893846035003662
Average time per epoch 0.7707471170425415 (for 500 epochs)
Epoch  377  time  0.8035831451416016
Average time per epoch 0.7723542833328247 (for 500 epochs)
Epoch  378  time  0.7957584857940674
Average time per epoch 0.7739458003044128 (for 500 epochs)
Epoch  379  time  0.8041341304779053
Average time per epoch 0.7755540685653687 (for 500 epochs)
Epoch  380  time  0.7840180397033691
Epoch  380  loss  1.013963008413501 correct 50
Average time per epoch 0.7771221046447754 (for 500 epochs)
Epoch  381  time  0.7888870239257812
Average time per epoch 0.7786998786926269 (for 500 epochs)
Epoch  382  time  0.8132138252258301
Average time per epoch 0.7803263063430786 (for 500 epochs)
Epoch  383  time  0.7990274429321289
Average time per epoch 0.7819243612289428 (for 500 epochs)
Epoch  384  time  0.8155477046966553
Average time per epoch 0.7835554566383361 (for 500 epochs)
Epoch  385  time  0.7894728183746338
Average time per epoch 0.7851344022750855 (for 500 epochs)
Epoch  386  time  0.7964439392089844
Average time per epoch 0.7867272901535034 (for 500 epochs)
Epoch  387  time  0.7951099872589111
Average time per epoch 0.7883175101280212 (for 500 epochs)
Epoch  388  time  0.775684118270874
Average time per epoch 0.789868878364563 (for 500 epochs)
Epoch  389  time  0.7682070732116699
Average time per epoch 0.7914052925109863 (for 500 epochs)
Epoch  390  time  0.8116626739501953
Epoch  390  loss  1.1401042670963732 correct 49
Average time per epoch 0.7930286178588867 (for 500 epochs)
Epoch  391  time  0.7896301746368408
Average time per epoch 0.7946078782081604 (for 500 epochs)
Epoch  392  time  0.796722412109375
Average time per epoch 0.7962013230323791 (for 500 epochs)
Epoch  393  time  0.8008840084075928
Average time per epoch 0.7978030910491943 (for 500 epochs)
Epoch  394  time  0.7730855941772461
Average time per epoch 0.7993492622375489 (for 500 epochs)
Epoch  395  time  0.82708740234375
Average time per epoch 0.8010034370422363 (for 500 epochs)
Epoch  396  time  0.8443841934204102
Average time per epoch 0.8026922054290772 (for 500 epochs)
Epoch  397  time  0.8062131404876709
Average time per epoch 0.8043046317100525 (for 500 epochs)
Epoch  398  time  0.7909016609191895
Average time per epoch 0.8058864350318908 (for 500 epochs)
Epoch  399  time  0.7838733196258545
Average time per epoch 0.8074541816711426 (for 500 epochs)
Epoch  400  time  0.8069331645965576
Epoch  400  loss  0.8688176731073957 correct 49
Average time per epoch 0.8090680480003357 (for 500 epochs)
Epoch  401  time  0.8038241863250732
Average time per epoch 0.8106756963729859 (for 500 epochs)
Epoch  402  time  0.7931771278381348
Average time per epoch 0.8122620506286621 (for 500 epochs)
Epoch  403  time  0.8264243602752686
Average time per epoch 0.8139148993492127 (for 500 epochs)
Epoch  404  time  0.8068172931671143
Average time per epoch 0.8155285339355469 (for 500 epochs)
Epoch  405  time  0.810654878616333
Average time per epoch 0.8171498436927795 (for 500 epochs)
Epoch  406  time  0.7888035774230957
Average time per epoch 0.8187274508476258 (for 500 epochs)
Epoch  407  time  0.801206111907959
Average time per epoch 0.8203298630714416 (for 500 epochs)
Epoch  408  time  0.7851099967956543
Average time per epoch 0.8219000830650329 (for 500 epochs)
Epoch  409  time  0.7881174087524414
Average time per epoch 0.8234763178825378 (for 500 epochs)
Epoch  410  time  0.8115246295928955
Epoch  410  loss  0.026454713868901948 correct 49
Average time per epoch 0.8250993671417236 (for 500 epochs)
Epoch  411  time  0.8192505836486816
Average time per epoch 0.826737868309021 (for 500 epochs)
Epoch  412  time  0.8036253452301025
Average time per epoch 0.8283451189994812 (for 500 epochs)
Epoch  413  time  0.7855072021484375
Average time per epoch 0.8299161334037781 (for 500 epochs)
Epoch  414  time  0.7929680347442627
Average time per epoch 0.8315020694732665 (for 500 epochs)
Epoch  415  time  0.8001189231872559
Average time per epoch 0.8331023073196411 (for 500 epochs)
Epoch  416  time  0.7835977077484131
Average time per epoch 0.834669502735138 (for 500 epochs)
Epoch  417  time  0.8057467937469482
Average time per epoch 0.8362809963226319 (for 500 epochs)
Epoch  418  time  0.8127491474151611
Average time per epoch 0.8379064946174621 (for 500 epochs)
Epoch  419  time  0.8026330471038818
Average time per epoch 0.8395117607116699 (for 500 epochs)
Epoch  420  time  0.7867898941040039
Epoch  420  loss  1.033841300014527 correct 49
Average time per epoch 0.8410853404998779 (for 500 epochs)
Epoch  421  time  0.7862825393676758
Average time per epoch 0.8426579055786133 (for 500 epochs)
Epoch  422  time  0.7780518531799316
Average time per epoch 0.8442140092849731 (for 500 epochs)
Epoch  423  time  0.7828109264373779
Average time per epoch 0.8457796311378479 (for 500 epochs)
Epoch  424  time  0.784149169921875
Average time per epoch 0.8473479294776917 (for 500 epochs)
Epoch  425  time  0.7843458652496338
Average time per epoch 0.8489166212081909 (for 500 epochs)
Epoch  426  time  0.7871992588043213
Average time per epoch 0.8504910197257995 (for 500 epochs)
Epoch  427  time  0.7953734397888184
Average time per epoch 0.8520817666053772 (for 500 epochs)
Epoch  428  time  0.7999916076660156
Average time per epoch 0.8536817498207092 (for 500 epochs)
Epoch  429  time  0.7954199314117432
Average time per epoch 0.8552725896835327 (for 500 epochs)
Epoch  430  time  0.7888245582580566
Epoch  430  loss  1.1151078507450434 correct 49
Average time per epoch 0.8568502388000488 (for 500 epochs)
Epoch  431  time  0.774965763092041
Average time per epoch 0.858400170326233 (for 500 epochs)
Epoch  432  time  0.7840368747711182
Average time per epoch 0.8599682440757751 (for 500 epochs)
Epoch  433  time  0.8327915668487549
Average time per epoch 0.8616338272094727 (for 500 epochs)
Epoch  434  time  0.8063399791717529
Average time per epoch 0.8632465071678161 (for 500 epochs)
Epoch  435  time  0.8062348365783691
Average time per epoch 0.8648589768409729 (for 500 epochs)
Epoch  436  time  1.4634182453155518
Average time per epoch 0.867785813331604 (for 500 epochs)
Epoch  437  time  7.596146583557129
Average time per epoch 0.8829781064987182 (for 500 epochs)
Epoch  438  time  3.6192402839660645
Average time per epoch 0.8902165870666504 (for 500 epochs)
Epoch  439  time  0.782515287399292
Average time per epoch 0.8917816176414489 (for 500 epochs)
Epoch  440  time  0.775968074798584
Epoch  440  loss  0.17546838554710342 correct 49
Average time per epoch 0.8933335537910462 (for 500 epochs)
Epoch  441  time  0.7827181816101074
Average time per epoch 0.8948989901542663 (for 500 epochs)
Epoch  442  time  0.7750475406646729
Average time per epoch 0.8964490852355957 (for 500 epochs)
Epoch  443  time  0.7843000888824463
Average time per epoch 0.8980176854133606 (for 500 epochs)
Epoch  444  time  0.7752363681793213
Average time per epoch 0.8995681581497192 (for 500 epochs)
Epoch  445  time  0.7741057872772217
Average time per epoch 0.9011163697242737 (for 500 epochs)
Epoch  446  time  0.804607629776001
Average time per epoch 0.9027255849838257 (for 500 epochs)
Epoch  447  time  0.8568496704101562
Average time per epoch 0.904439284324646 (for 500 epochs)
Epoch  448  time  0.7923815250396729
Average time per epoch 0.9060240473747253 (for 500 epochs)
Epoch  449  time  0.7810060977935791
Average time per epoch 0.9075860595703125 (for 500 epochs)
Epoch  450  time  0.7966456413269043
Epoch  450  loss  1.2369094003155183 correct 49
Average time per epoch 0.9091793508529663 (for 500 epochs)
Epoch  451  time  0.795051097869873
Average time per epoch 0.910769453048706 (for 500 epochs)
Epoch  452  time  0.7973756790161133
Average time per epoch 0.9123642044067383 (for 500 epochs)
Epoch  453  time  0.7870211601257324
Average time per epoch 0.9139382467269898 (for 500 epochs)
Epoch  454  time  0.8012628555297852
Average time per epoch 0.9155407724380493 (for 500 epochs)
Epoch  455  time  0.7890236377716064
Average time per epoch 0.9171188197135925 (for 500 epochs)
Epoch  456  time  0.7935609817504883
Average time per epoch 0.9187059416770935 (for 500 epochs)
Epoch  457  time  0.8028860092163086
Average time per epoch 0.9203117136955261 (for 500 epochs)
Epoch  458  time  0.8059618473052979
Average time per epoch 0.9219236373901367 (for 500 epochs)
Epoch  459  time  0.8469996452331543
Average time per epoch 0.923617636680603 (for 500 epochs)
Epoch  460  time  0.7946712970733643
Epoch  460  loss  0.10235862143459412 correct 49
Average time per epoch 0.9252069792747497 (for 500 epochs)
Epoch  461  time  0.7959434986114502
Average time per epoch 0.9267988662719726 (for 500 epochs)
Epoch  462  time  0.8229975700378418
Average time per epoch 0.9284448614120483 (for 500 epochs)
Epoch  463  time  0.7687246799468994
Average time per epoch 0.9299823107719422 (for 500 epochs)
Epoch  464  time  0.8064897060394287
Average time per epoch 0.931595290184021 (for 500 epochs)
Epoch  465  time  0.7833008766174316
Average time per epoch 0.9331618919372558 (for 500 epochs)
Epoch  466  time  0.7895951271057129
Average time per epoch 0.9347410821914672 (for 500 epochs)
Epoch  467  time  0.7929372787475586
Average time per epoch 0.9363269567489624 (for 500 epochs)
Epoch  468  time  0.7961959838867188
Average time per epoch 0.9379193487167359 (for 500 epochs)
Epoch  469  time  0.7874584197998047
Average time per epoch 0.9394942655563354 (for 500 epochs)
Epoch  470  time  0.7956089973449707
Epoch  470  loss  0.01116434688404379 correct 49
Average time per epoch 0.9410854835510254 (for 500 epochs)
Epoch  471  time  0.7804043292999268
Average time per epoch 0.9426462922096253 (for 500 epochs)
Epoch  472  time  0.7904036045074463
Average time per epoch 0.9442270994186401 (for 500 epochs)
Epoch  473  time  0.7872915267944336
Average time per epoch 0.945801682472229 (for 500 epochs)
Epoch  474  time  0.7861514091491699
Average time per epoch 0.9473739852905273 (for 500 epochs)
Epoch  475  time  0.7758710384368896
Average time per epoch 0.9489257273674011 (for 500 epochs)
Epoch  476  time  0.7628140449523926
Average time per epoch 0.9504513554573059 (for 500 epochs)
Epoch  477  time  0.7797880172729492
Average time per epoch 0.9520109314918518 (for 500 epochs)
Epoch  478  time  0.8037278652191162
Average time per epoch 0.9536183872222901 (for 500 epochs)
Epoch  479  time  0.7919180393218994
Average time per epoch 0.9552022233009339 (for 500 epochs)
Epoch  480  time  2.2780799865722656
Epoch  480  loss  0.9629708281730677 correct 49
Average time per epoch 0.9597583832740784 (for 500 epochs)
Epoch  481  time  5.953208923339844
Average time per epoch 0.971664801120758 (for 500 epochs)
Epoch  482  time  3.8191325664520264
Average time per epoch 0.9793030662536621 (for 500 epochs)
Epoch  483  time  0.7692890167236328
Average time per epoch 0.9808416442871094 (for 500 epochs)
Epoch  484  time  0.8388442993164062
Average time per epoch 0.9825193328857422 (for 500 epochs)
Epoch  485  time  0.8102478981018066
Average time per epoch 0.9841398286819458 (for 500 epochs)
Epoch  486  time  0.7744402885437012
Average time per epoch 0.9856887092590332 (for 500 epochs)
Epoch  487  time  0.7963778972625732
Average time per epoch 0.9872814650535584 (for 500 epochs)
Epoch  488  time  0.7660350799560547
Average time per epoch 0.9888135352134705 (for 500 epochs)
Epoch  489  time  0.7767925262451172
Average time per epoch 0.9903671202659607 (for 500 epochs)
Epoch  490  time  0.7744903564453125
Epoch  490  loss  0.00700473603341315 correct 49
Average time per epoch 0.9919161009788513 (for 500 epochs)
Epoch  491  time  0.7714312076568604
Average time per epoch 0.9934589633941651 (for 500 epochs)
Epoch  492  time  0.7748653888702393
Average time per epoch 0.9950086941719055 (for 500 epochs)
Epoch  493  time  0.7876391410827637
Average time per epoch 0.996583972454071 (for 500 epochs)
Epoch  494  time  0.769033670425415
Average time per epoch 0.9981220397949219 (for 500 epochs)
Epoch  495  time  0.798914909362793
Average time per epoch 0.9997198696136474 (for 500 epochs)
Epoch  496  time  0.7767102718353271
Average time per epoch 1.001273290157318 (for 500 epochs)
Epoch  497  time  0.7708477973937988
Average time per epoch 1.0028149857521058 (for 500 epochs)
Epoch  498  time  0.805117130279541
Average time per epoch 1.0044252200126649 (for 500 epochs)
Epoch  499  time  0.8129456043243408
Average time per epoch 1.0060511112213135 (for 500 epochs)
```

# GPU Simple Dataset:
```
Epoch  0  loss  4.931410659775688 correct 32
Average time per epoch 0.00913517951965332 (for 500 epochs)
Epoch  1  time  2.047029733657837
Average time per epoch 0.013229238986968993 (for 500 epochs)
Epoch  2  time  1.9726250171661377
Average time per epoch 0.01717448902130127 (for 500 epochs)
Epoch  3  time  1.9437272548675537
Average time per epoch 0.021061943531036378 (for 500 epochs)
Epoch  4  time  1.9641282558441162
Average time per epoch 0.02499020004272461 (for 500 epochs)
Epoch  5  time  2.0343081951141357
Average time per epoch 0.02905881643295288 (for 500 epochs)
Epoch  6  time  1.9989039897918701
Average time per epoch 0.03305662441253662 (for 500 epochs)
Epoch  7  time  1.9676482677459717
Average time per epoch 0.03699192094802856 (for 500 epochs)
Epoch  8  time  2.0960497856140137
Average time per epoch 0.04118402051925659 (for 500 epochs)
Epoch  9  time  2.0372560024261475
Average time per epoch 0.045258532524108885 (for 500 epochs)
Epoch  10  time  2.0064995288848877
Epoch  10  loss  1.9420022718389889 correct 47
Average time per epoch 0.049271531581878664 (for 500 epochs)
Epoch  11  time  2.037872791290283
Average time per epoch 0.05334727716445923 (for 500 epochs)
Epoch  12  time  1.9887478351593018
Average time per epoch 0.05732477283477783 (for 500 epochs)
Epoch  13  time  1.9816372394561768
Average time per epoch 0.06128804731369018 (for 500 epochs)
Epoch  14  time  1.9818031787872314
Average time per epoch 0.06525165367126465 (for 500 epochs)
Epoch  15  time  2.059206485748291
Average time per epoch 0.06937006664276123 (for 500 epochs)
Epoch  16  time  1.945791244506836
Average time per epoch 0.07326164913177491 (for 500 epochs)
Epoch  17  time  1.9475831985473633
Average time per epoch 0.07715681552886963 (for 500 epochs)
Epoch  18  time  2.0313918590545654
Average time per epoch 0.08121959924697876 (for 500 epochs)
Epoch  19  time  1.9870271682739258
Average time per epoch 0.08519365358352661 (for 500 epochs)
Epoch  20  time  1.9878170490264893
Epoch  20  loss  1.933571214425776 correct 50
Average time per epoch 0.08916928768157958 (for 500 epochs)
Epoch  21  time  2.2135720252990723
Average time per epoch 0.09359643173217773 (for 500 epochs)
Epoch  22  time  3.377200126647949
Average time per epoch 0.10035083198547363 (for 500 epochs)
Epoch  23  time  3.4652953147888184
Average time per epoch 0.10728142261505128 (for 500 epochs)
Epoch  24  time  3.43135404586792
Average time per epoch 0.11414413070678711 (for 500 epochs)
Epoch  25  time  2.265873670578003
Average time per epoch 0.11867587804794312 (for 500 epochs)
Epoch  26  time  1.9752821922302246
Average time per epoch 0.12262644243240356 (for 500 epochs)
Epoch  27  time  2.0061824321746826
Average time per epoch 0.12663880729675292 (for 500 epochs)
Epoch  28  time  1.971580982208252
Average time per epoch 0.13058196926116944 (for 500 epochs)
Epoch  29  time  2.0261342525482178
Average time per epoch 0.13463423776626587 (for 500 epochs)
Epoch  30  time  1.9502334594726562
Epoch  30  loss  1.168722813353001 correct 50
Average time per epoch 0.13853470468521117 (for 500 epochs)
Epoch  31  time  1.9617738723754883
Average time per epoch 0.14245825242996216 (for 500 epochs)
Epoch  32  time  2.03132963180542
Average time per epoch 0.146520911693573 (for 500 epochs)
Epoch  33  time  1.9947710037231445
Average time per epoch 0.1505104537010193 (for 500 epochs)
Epoch  34  time  1.985888957977295
Average time per epoch 0.1544822316169739 (for 500 epochs)
Epoch  35  time  2.0256059169769287
Average time per epoch 0.15853344345092774 (for 500 epochs)
Epoch  36  time  1.9327166080474854
Average time per epoch 0.1623988766670227 (for 500 epochs)
Epoch  37  time  2.0008342266082764
Average time per epoch 0.16640054512023925 (for 500 epochs)
Epoch  38  time  1.9406547546386719
Average time per epoch 0.1702818546295166 (for 500 epochs)
Epoch  39  time  2.0181965827941895
Average time per epoch 0.174318247795105 (for 500 epochs)
Epoch  40  time  1.9462919235229492
Epoch  40  loss  2.976186532018095 correct 49
Average time per epoch 0.17821083164215087 (for 500 epochs)
Epoch  41  time  1.9417123794555664
Average time per epoch 0.182094256401062 (for 500 epochs)
Epoch  42  time  2.0286195278167725
Average time per epoch 0.18615149545669557 (for 500 epochs)
Epoch  43  time  1.9671833515167236
Average time per epoch 0.190085862159729 (for 500 epochs)
Epoch  44  time  1.9478464126586914
Average time per epoch 0.19398155498504638 (for 500 epochs)
Epoch  45  time  2.0142385959625244
Average time per epoch 0.19801003217697144 (for 500 epochs)
Epoch  46  time  1.9524850845336914
Average time per epoch 0.20191500234603882 (for 500 epochs)
Epoch  47  time  1.973036766052246
Average time per epoch 0.2058610758781433 (for 500 epochs)
Epoch  48  time  1.9660985469818115
Average time per epoch 0.20979327297210693 (for 500 epochs)
Epoch  49  time  2.045391321182251
Average time per epoch 0.21388405561447144 (for 500 epochs)
Epoch  50  time  1.9465668201446533
Epoch  50  loss  0.5942385634823224 correct 50
Average time per epoch 0.21777718925476075 (for 500 epochs)
Epoch  51  time  1.963571310043335
Average time per epoch 0.22170433187484742 (for 500 epochs)
Epoch  52  time  2.0599024295806885
Average time per epoch 0.22582413673400878 (for 500 epochs)
Epoch  53  time  1.9938709735870361
Average time per epoch 0.22981187868118286 (for 500 epochs)
Epoch  54  time  2.6053664684295654
Average time per epoch 0.235022611618042 (for 500 epochs)
Epoch  55  time  3.3833985328674316
Average time per epoch 0.24178940868377685 (for 500 epochs)
Epoch  56  time  3.6048383712768555
Average time per epoch 0.24899908542633056 (for 500 epochs)
Epoch  57  time  3.13232421875
Average time per epoch 0.25526373386383056 (for 500 epochs)
Epoch  58  time  1.9624645709991455
Average time per epoch 0.2591886630058289 (for 500 epochs)
Epoch  59  time  2.0396814346313477
Average time per epoch 0.26326802587509157 (for 500 epochs)
Epoch  60  time  1.9607336521148682
Epoch  60  loss  1.0305086067326452 correct 50
Average time per epoch 0.2671894931793213 (for 500 epochs)
Epoch  61  time  1.9856016635894775
Average time per epoch 0.27116069650650027 (for 500 epochs)
Epoch  62  time  2.0344507694244385
Average time per epoch 0.27522959804534913 (for 500 epochs)
Epoch  63  time  1.9603657722473145
Average time per epoch 0.27915032958984376 (for 500 epochs)
Epoch  64  time  1.982520341873169
Average time per epoch 0.2831153702735901 (for 500 epochs)
Epoch  65  time  1.9967966079711914
Average time per epoch 0.28710896348953246 (for 500 epochs)
Epoch  66  time  2.0663504600524902
Average time per epoch 0.29124166440963745 (for 500 epochs)
Epoch  67  time  1.9504976272583008
Average time per epoch 0.29514265966415404 (for 500 epochs)
Epoch  68  time  1.9619362354278564
Average time per epoch 0.2990665321350098 (for 500 epochs)
Epoch  69  time  2.014364004135132
Average time per epoch 0.30309526014328003 (for 500 epochs)
Epoch  70  time  1.9850714206695557
Epoch  70  loss  0.7592328291479091 correct 50
Average time per epoch 0.30706540298461915 (for 500 epochs)
Epoch  71  time  1.9849603176116943
Average time per epoch 0.31103532361984254 (for 500 epochs)
Epoch  72  time  2.0396792888641357
Average time per epoch 0.31511468219757083 (for 500 epochs)
Epoch  73  time  1.9809143543243408
Average time per epoch 0.31907651090621947 (for 500 epochs)
Epoch  74  time  1.9394593238830566
Average time per epoch 0.3229554295539856 (for 500 epochs)
Epoch  75  time  1.9444572925567627
Average time per epoch 0.32684434413909913 (for 500 epochs)
Epoch  76  time  2.0049171447753906
Average time per epoch 0.3308541784286499 (for 500 epochs)
Epoch  77  time  1.9316370487213135
Average time per epoch 0.33471745252609253 (for 500 epochs)
Epoch  78  time  1.9374125003814697
Average time per epoch 0.33859227752685545 (for 500 epochs)
Epoch  79  time  2.022282123565674
Average time per epoch 0.3426368417739868 (for 500 epochs)
Epoch  80  time  2.010890483856201
Epoch  80  loss  0.28876765482126493 correct 50
Average time per epoch 0.34665862274169923 (for 500 epochs)
Epoch  81  time  2.012383222579956
Average time per epoch 0.35068338918685915 (for 500 epochs)
Epoch  82  time  1.9491322040557861
Average time per epoch 0.3545816535949707 (for 500 epochs)
Epoch  83  time  2.003779411315918
Average time per epoch 0.35858921241760255 (for 500 epochs)
Epoch  84  time  1.9599647521972656
Average time per epoch 0.36250914192199707 (for 500 epochs)
Epoch  85  time  2.82364559173584
Average time per epoch 0.36815643310546875 (for 500 epochs)
Epoch  86  time  3.461287498474121
Average time per epoch 0.375079008102417 (for 500 epochs)
Epoch  87  time  3.342003107070923
Average time per epoch 0.38176301431655885 (for 500 epochs)
Epoch  88  time  3.5216972827911377
Average time per epoch 0.38880640888214113 (for 500 epochs)
Epoch  89  time  3.503875732421875
Average time per epoch 0.39581416034698486 (for 500 epochs)
Epoch  90  time  2.3979716300964355
Epoch  90  loss  0.09457646999281824 correct 50
Average time per epoch 0.40061010360717775 (for 500 epochs)
Epoch  91  time  1.9665277004241943
Average time per epoch 0.4045431590080261 (for 500 epochs)
Epoch  92  time  2.0633702278137207
Average time per epoch 0.40866989946365356 (for 500 epochs)
Epoch  93  time  2.0585758686065674
Average time per epoch 0.4127870512008667 (for 500 epochs)
Epoch  94  time  1.9306998252868652
Average time per epoch 0.4166484508514404 (for 500 epochs)
Epoch  95  time  1.9533584117889404
Average time per epoch 0.4205551676750183 (for 500 epochs)
Epoch  96  time  2.015031337738037
Average time per epoch 0.4245852303504944 (for 500 epochs)
Epoch  97  time  1.97882080078125
Average time per epoch 0.42854287195205687 (for 500 epochs)
Epoch  98  time  1.956413984298706
Average time per epoch 0.4324556999206543 (for 500 epochs)
Epoch  99  time  1.962855577468872
Average time per epoch 0.43638141107559203 (for 500 epochs)
Epoch  100  time  2.0462679862976074
Epoch  100  loss  0.33842348705312697 correct 50
Average time per epoch 0.44047394704818726 (for 500 epochs)
Epoch  101  time  1.9783782958984375
Average time per epoch 0.4444307036399841 (for 500 epochs)
Epoch  102  time  1.9637746810913086
Average time per epoch 0.44835825300216675 (for 500 epochs)
Epoch  103  time  2.0542116165161133
Average time per epoch 0.452466676235199 (for 500 epochs)
Epoch  104  time  2.032083034515381
Average time per epoch 0.45653084230422974 (for 500 epochs)
Epoch  105  time  1.9425420761108398
Average time per epoch 0.4604159264564514 (for 500 epochs)
Epoch  106  time  2.0727946758270264
Average time per epoch 0.46456151580810545 (for 500 epochs)
Epoch  107  time  2.0184166431427
Average time per epoch 0.46859834909439085 (for 500 epochs)
Epoch  108  time  1.9544358253479004
Average time per epoch 0.47250722074508666 (for 500 epochs)
Epoch  109  time  1.942117691040039
Average time per epoch 0.47639145612716677 (for 500 epochs)
Epoch  110  time  2.0167112350463867
Epoch  110  loss  0.34499524891749866 correct 50
Average time per epoch 0.4804248785972595 (for 500 epochs)
Epoch  111  time  1.9696600437164307
Average time per epoch 0.4843641986846924 (for 500 epochs)
Epoch  112  time  2.6470861434936523
Average time per epoch 0.4896583709716797 (for 500 epochs)
Epoch  113  time  3.472295045852661
Average time per epoch 0.49660296106338503 (for 500 epochs)
Epoch  114  time  3.3559834957122803
Average time per epoch 0.5033149280548096 (for 500 epochs)
Epoch  115  time  3.2130775451660156
Average time per epoch 0.5097410831451415 (for 500 epochs)
Epoch  116  time  2.039665460586548
Average time per epoch 0.5138204140663147 (for 500 epochs)
Epoch  117  time  2.3178694248199463
Average time per epoch 0.5184561529159546 (for 500 epochs)
Epoch  118  time  3.3392229080200195
Average time per epoch 0.5251345987319946 (for 500 epochs)
Epoch  119  time  3.467930316925049
Average time per epoch 0.5320704593658447 (for 500 epochs)
Epoch  120  time  3.524590253829956
Epoch  120  loss  0.08501422370661185 correct 50
Average time per epoch 0.5391196398735046 (for 500 epochs)
Epoch  121  time  2.073554039001465
Average time per epoch 0.5432667479515075 (for 500 epochs)
Epoch  122  time  1.9517958164215088
Average time per epoch 0.5471703395843506 (for 500 epochs)
Epoch  123  time  2.0252602100372314
Average time per epoch 0.551220860004425 (for 500 epochs)
Epoch  124  time  1.9476490020751953
Average time per epoch 0.5551161580085754 (for 500 epochs)
Epoch  125  time  1.9422886371612549
Average time per epoch 0.559000735282898 (for 500 epochs)
Epoch  126  time  1.9522907733917236
Average time per epoch 0.5629053168296814 (for 500 epochs)
Epoch  127  time  2.042407989501953
Average time per epoch 0.5669901328086853 (for 500 epochs)
Epoch  128  time  1.964174747467041
Average time per epoch 0.5709184823036194 (for 500 epochs)
Epoch  129  time  1.9703309535980225
Average time per epoch 0.5748591442108154 (for 500 epochs)
Epoch  130  time  2.0220909118652344
Epoch  130  loss  0.7399923741732227 correct 50
Average time per epoch 0.5789033260345459 (for 500 epochs)
Epoch  131  time  1.9337129592895508
Average time per epoch 0.582770751953125 (for 500 epochs)
Epoch  132  time  2.0030510425567627
Average time per epoch 0.5867768540382385 (for 500 epochs)
Epoch  133  time  2.0649006366729736
Average time per epoch 0.5909066553115845 (for 500 epochs)
Epoch  134  time  1.978234052658081
Average time per epoch 0.5948631234169006 (for 500 epochs)
Epoch  135  time  1.985783576965332
Average time per epoch 0.5988346905708313 (for 500 epochs)
Epoch  136  time  1.9717504978179932
Average time per epoch 0.6027781915664673 (for 500 epochs)
Epoch  137  time  2.0266873836517334
Average time per epoch 0.6068315663337708 (for 500 epochs)
Epoch  138  time  2.0033977031707764
Average time per epoch 0.6108383617401123 (for 500 epochs)
Epoch  139  time  1.9389233589172363
Average time per epoch 0.6147162084579467 (for 500 epochs)
Epoch  140  time  2.006843090057373
Epoch  140  loss  0.3228284904173023 correct 50
Average time per epoch 0.6187298946380615 (for 500 epochs)
Epoch  141  time  1.9524636268615723
Average time per epoch 0.6226348218917847 (for 500 epochs)
Epoch  142  time  1.9437434673309326
Average time per epoch 0.6265223088264466 (for 500 epochs)
Epoch  143  time  1.9948625564575195
Average time per epoch 0.6305120339393616 (for 500 epochs)
Epoch  144  time  2.0156092643737793
Average time per epoch 0.6345432524681092 (for 500 epochs)
Epoch  145  time  1.9543249607086182
Average time per epoch 0.6384519023895263 (for 500 epochs)
Epoch  146  time  1.9593520164489746
Average time per epoch 0.6423706064224243 (for 500 epochs)
Epoch  147  time  2.0165207386016846
Average time per epoch 0.6464036478996277 (for 500 epochs)
Epoch  148  time  1.9648268222808838
Average time per epoch 0.6503333015441894 (for 500 epochs)
Epoch  149  time  1.906872272491455
Average time per epoch 0.6541470460891724 (for 500 epochs)
Epoch  150  time  2.0504918098449707
Epoch  150  loss  0.3761509277561908 correct 50
Average time per epoch 0.6582480297088623 (for 500 epochs)
Epoch  151  time  2.76056170463562
Average time per epoch 0.6637691531181336 (for 500 epochs)
Epoch  152  time  3.430246353149414
Average time per epoch 0.6706296458244324 (for 500 epochs)
Epoch  153  time  3.4164552688598633
Average time per epoch 0.6774625563621521 (for 500 epochs)
Epoch  154  time  3.0688867568969727
Average time per epoch 0.6836003298759461 (for 500 epochs)
Epoch  155  time  1.9725699424743652
Average time per epoch 0.6875454697608948 (for 500 epochs)
Epoch  156  time  1.9524428844451904
Average time per epoch 0.6914503555297852 (for 500 epochs)
Epoch  157  time  2.011343240737915
Average time per epoch 0.695473042011261 (for 500 epochs)
Epoch  158  time  1.9343769550323486
Average time per epoch 0.6993417959213257 (for 500 epochs)
Epoch  159  time  1.9584364891052246
Average time per epoch 0.7032586688995361 (for 500 epochs)
Epoch  160  time  2.022784948348999
Epoch  160  loss  0.06115284743084999 correct 50
Average time per epoch 0.7073042387962342 (for 500 epochs)
Epoch  161  time  1.9752612113952637
Average time per epoch 0.7112547612190246 (for 500 epochs)
Epoch  162  time  1.9230108261108398
Average time per epoch 0.7151007828712463 (for 500 epochs)
Epoch  163  time  1.9667584896087646
Average time per epoch 0.7190342998504639 (for 500 epochs)
Epoch  164  time  2.089970350265503
Average time per epoch 0.7232142405509949 (for 500 epochs)
Epoch  165  time  1.975949764251709
Average time per epoch 0.7271661400794983 (for 500 epochs)
Epoch  166  time  1.9817533493041992
Average time per epoch 0.7311296467781067 (for 500 epochs)
Epoch  167  time  2.007248640060425
Average time per epoch 0.7351441440582276 (for 500 epochs)
Epoch  168  time  1.9285087585449219
Average time per epoch 0.7390011615753174 (for 500 epochs)
Epoch  169  time  1.9369726181030273
Average time per epoch 0.7428751068115235 (for 500 epochs)
Epoch  170  time  1.9633710384368896
Epoch  170  loss  0.6975953926356199 correct 50
Average time per epoch 0.7468018488883972 (for 500 epochs)
Epoch  171  time  2.058614492416382
Average time per epoch 0.75091907787323 (for 500 epochs)
Epoch  172  time  1.9736690521240234
Average time per epoch 0.7548664159774781 (for 500 epochs)
Epoch  173  time  1.9795832633972168
Average time per epoch 0.7588255825042725 (for 500 epochs)
Epoch  174  time  2.009312152862549
Average time per epoch 0.7628442068099975 (for 500 epochs)
Epoch  175  time  1.9763069152832031
Average time per epoch 0.766796820640564 (for 500 epochs)
Epoch  176  time  1.9604542255401611
Average time per epoch 0.7707177290916443 (for 500 epochs)
Epoch  177  time  2.0566015243530273
Average time per epoch 0.7748309321403504 (for 500 epochs)
Epoch  178  time  1.999631643295288
Average time per epoch 0.7788301954269409 (for 500 epochs)
Epoch  179  time  1.939138650894165
Average time per epoch 0.7827084727287292 (for 500 epochs)
Epoch  180  time  1.9552786350250244
Epoch  180  loss  0.157553723277251 correct 50
Average time per epoch 0.7866190299987793 (for 500 epochs)
Epoch  181  time  2.006629228591919
Average time per epoch 0.7906322884559631 (for 500 epochs)
Epoch  182  time  1.9598057270050049
Average time per epoch 0.7945518999099731 (for 500 epochs)
Epoch  183  time  1.959022045135498
Average time per epoch 0.7984699440002442 (for 500 epochs)
Epoch  184  time  2.0125844478607178
Average time per epoch 0.8024951128959655 (for 500 epochs)
Epoch  185  time  2.5015811920166016
Average time per epoch 0.8074982752799987 (for 500 epochs)
Epoch  186  time  3.3768911361694336
Average time per epoch 0.8142520575523376 (for 500 epochs)
Epoch  187  time  3.4698009490966797
Average time per epoch 0.821191659450531 (for 500 epochs)
Epoch  188  time  3.305845022201538
Average time per epoch 0.8278033494949341 (for 500 epochs)
Epoch  189  time  1.9346437454223633
Average time per epoch 0.8316726369857788 (for 500 epochs)
Epoch  190  time  1.9538815021514893
Epoch  190  loss  0.007817611539490357 correct 50
Average time per epoch 0.8355803999900818 (for 500 epochs)
Epoch  191  time  2.0279057025909424
Average time per epoch 0.8396362113952637 (for 500 epochs)
Epoch  192  time  1.9359350204467773
Average time per epoch 0.8435080814361572 (for 500 epochs)
Epoch  193  time  1.9536848068237305
Average time per epoch 0.8474154510498046 (for 500 epochs)
Epoch  194  time  2.0204288959503174
Average time per epoch 0.8514563088417053 (for 500 epochs)
Epoch  195  time  1.9038360118865967
Average time per epoch 0.8552639808654785 (for 500 epochs)
Epoch  196  time  1.9369375705718994
Average time per epoch 0.8591378560066223 (for 500 epochs)
Epoch  197  time  1.9468488693237305
Average time per epoch 0.8630315537452697 (for 500 epochs)
Epoch  198  time  2.0002079010009766
Average time per epoch 0.8670319695472717 (for 500 epochs)
Epoch  199  time  1.9628922939300537
Average time per epoch 0.8709577541351319 (for 500 epochs)
Epoch  200  time  1.95381760597229
Epoch  200  loss  0.2663892262094196 correct 50
Average time per epoch 0.8748653893470764 (for 500 epochs)
Epoch  201  time  2.027412176132202
Average time per epoch 0.8789202136993408 (for 500 epochs)
Epoch  202  time  1.9438707828521729
Average time per epoch 0.8828079552650452 (for 500 epochs)
Epoch  203  time  1.9710862636566162
Average time per epoch 0.8867501277923584 (for 500 epochs)
Epoch  204  time  2.040802478790283
Average time per epoch 0.890831732749939 (for 500 epochs)
Epoch  205  time  1.9906206130981445
Average time per epoch 0.8948129739761352 (for 500 epochs)
Epoch  206  time  1.9730236530303955
Average time per epoch 0.8987590212821961 (for 500 epochs)
Epoch  207  time  1.9458656311035156
Average time per epoch 0.9026507525444031 (for 500 epochs)
Epoch  208  time  2.0129570960998535
Average time per epoch 0.9066766667366027 (for 500 epochs)
Epoch  209  time  1.9360520839691162
Average time per epoch 0.910548770904541 (for 500 epochs)
Epoch  210  time  1.9726018905639648
Epoch  210  loss  0.27871135442305967 correct 50
Average time per epoch 0.914493974685669 (for 500 epochs)
Epoch  211  time  2.0388593673706055
Average time per epoch 0.9185716934204101 (for 500 epochs)
Epoch  212  time  1.9434895515441895
Average time per epoch 0.9224586725234986 (for 500 epochs)
Epoch  213  time  1.9591400623321533
Average time per epoch 0.9263769526481629 (for 500 epochs)
Epoch  214  time  1.9718055725097656
Average time per epoch 0.9303205637931824 (for 500 epochs)
Epoch  215  time  2.0393786430358887
Average time per epoch 0.9343993210792542 (for 500 epochs)
Epoch  216  time  1.9330050945281982
Average time per epoch 0.9382653312683106 (for 500 epochs)
Epoch  217  time  1.9385359287261963
Average time per epoch 0.9421424031257629 (for 500 epochs)
Epoch  218  time  2.208899736404419
Average time per epoch 0.9465602025985718 (for 500 epochs)
Epoch  219  time  3.233752965927124
Average time per epoch 0.9530277085304261 (for 500 epochs)
Epoch  220  time  3.357938289642334
Epoch  220  loss  0.45740070588630233 correct 50
Average time per epoch 0.9597435851097107 (for 500 epochs)
Epoch  221  time  3.5697479248046875
Average time per epoch 0.9668830809593201 (for 500 epochs)
Epoch  222  time  3.4125170707702637
Average time per epoch 0.9737081151008606 (for 500 epochs)
Epoch  223  time  2.489349365234375
Average time per epoch 0.9786868138313294 (for 500 epochs)
Epoch  224  time  1.9535157680511475
Average time per epoch 0.9825938453674317 (for 500 epochs)
Epoch  225  time  2.002713441848755
Average time per epoch 0.9865992722511292 (for 500 epochs)
Epoch  226  time  1.969810962677002
Average time per epoch 0.9905388941764831 (for 500 epochs)
Epoch  227  time  1.9439404010772705
Average time per epoch 0.9944267749786377 (for 500 epochs)
Epoch  228  time  2.0193676948547363
Average time per epoch 0.9984655103683472 (for 500 epochs)
Epoch  229  time  1.9726958274841309
Average time per epoch 1.0024109020233154 (for 500 epochs)
Epoch  230  time  1.9725210666656494
Epoch  230  loss  0.09861225457505124 correct 50
Average time per epoch 1.0063559441566468 (for 500 epochs)
Epoch  231  time  1.9623162746429443
Average time per epoch 1.0102805767059326 (for 500 epochs)
Epoch  232  time  2.0331552028656006
Average time per epoch 1.0143468871116639 (for 500 epochs)
Epoch  233  time  1.996340036392212
Average time per epoch 1.0183395671844482 (for 500 epochs)
Epoch  234  time  1.9880406856536865
Average time per epoch 1.0223156485557556 (for 500 epochs)
Epoch  235  time  2.0215415954589844
Average time per epoch 1.0263587317466736 (for 500 epochs)
Epoch  236  time  1.970000982284546
Average time per epoch 1.0302987337112426 (for 500 epochs)
Epoch  237  time  1.9906830787658691
Average time per epoch 1.0342800998687744 (for 500 epochs)
Epoch  238  time  2.0124597549438477
Average time per epoch 1.038305019378662 (for 500 epochs)
Epoch  239  time  2.0180537700653076
Average time per epoch 1.0423411269187928 (for 500 epochs)
Epoch  240  time  1.960200548171997
Epoch  240  loss  0.3096967862821653 correct 50
Average time per epoch 1.0462615280151366 (for 500 epochs)
Epoch  241  time  1.9581148624420166
Average time per epoch 1.0501777577400206 (for 500 epochs)
Epoch  242  time  2.1420466899871826
Average time per epoch 1.054461851119995 (for 500 epochs)
Epoch  243  time  1.9537708759307861
Average time per epoch 1.0583693928718567 (for 500 epochs)
Epoch  244  time  1.9435410499572754
Average time per epoch 1.0622564749717713 (for 500 epochs)
Epoch  245  time  2.0222718715667725
Average time per epoch 1.0663010187149047 (for 500 epochs)
Epoch  246  time  1.9497878551483154
Average time per epoch 1.0702005944252013 (for 500 epochs)
Epoch  247  time  2.7439496517181396
Average time per epoch 1.0756884937286377 (for 500 epochs)
Epoch  248  time  3.4133002758026123
Average time per epoch 1.0825150942802428 (for 500 epochs)
Epoch  249  time  3.5395543575286865
Average time per epoch 1.0895942029953003 (for 500 epochs)
Epoch  250  time  3.342632532119751
Epoch  250  loss  0.15406323773456304 correct 50
Average time per epoch 1.0962794680595398 (for 500 epochs)
Epoch  251  time  3.374629020690918
Average time per epoch 1.1030287261009217 (for 500 epochs)
Epoch  252  time  3.465977191925049
Average time per epoch 1.1099606804847717 (for 500 epochs)
Epoch  253  time  2.433875560760498
Average time per epoch 1.1148284316062926 (for 500 epochs)
Epoch  254  time  1.9348034858703613
Average time per epoch 1.1186980385780334 (for 500 epochs)
Epoch  255  time  2.023557186126709
Average time per epoch 1.1227451529502868 (for 500 epochs)
Epoch  256  time  1.9564156532287598
Average time per epoch 1.1266579842567443 (for 500 epochs)
Epoch  257  time  1.9695420265197754
Average time per epoch 1.130597068309784 (for 500 epochs)
Epoch  258  time  1.9474115371704102
Average time per epoch 1.1344918913841247 (for 500 epochs)
Epoch  259  time  1.9992311000823975
Average time per epoch 1.1384903535842896 (for 500 epochs)
Epoch  260  time  2.0297741889953613
Epoch  260  loss  0.007939824041799546 correct 50
Average time per epoch 1.1425499019622802 (for 500 epochs)
Epoch  261  time  1.9647376537322998
Average time per epoch 1.1464793772697448 (for 500 epochs)
Epoch  262  time  1.9975199699401855
Average time per epoch 1.1504744172096253 (for 500 epochs)
Epoch  263  time  1.9435181617736816
Average time per epoch 1.1543614535331725 (for 500 epochs)
Epoch  264  time  1.9411194324493408
Average time per epoch 1.1582436923980712 (for 500 epochs)
Epoch  265  time  1.9287679195404053
Average time per epoch 1.1621012282371521 (for 500 epochs)
Epoch  266  time  2.029001235961914
Average time per epoch 1.166159230709076 (for 500 epochs)
Epoch  267  time  1.9691364765167236
Average time per epoch 1.1700975036621093 (for 500 epochs)
Epoch  268  time  1.9561207294464111
Average time per epoch 1.174009745121002 (for 500 epochs)
Epoch  269  time  2.0191726684570312
Average time per epoch 1.1780480904579163 (for 500 epochs)
Epoch  270  time  1.95290207862854
Epoch  270  loss  0.25001995883276634 correct 50
Average time per epoch 1.1819538946151733 (for 500 epochs)
Epoch  271  time  1.9957079887390137
Average time per epoch 1.1859453105926514 (for 500 epochs)
Epoch  272  time  2.015596628189087
Average time per epoch 1.1899765038490295 (for 500 epochs)
Epoch  273  time  1.9287419319152832
Average time per epoch 1.19383398771286 (for 500 epochs)
Epoch  274  time  1.9235897064208984
Average time per epoch 1.197681167125702 (for 500 epochs)
Epoch  275  time  1.9483006000518799
Average time per epoch 1.2015777683258058 (for 500 epochs)
Epoch  276  time  2.0211033821105957
Average time per epoch 1.205619975090027 (for 500 epochs)
Epoch  277  time  1.9357542991638184
Average time per epoch 1.2094914836883546 (for 500 epochs)
Epoch  278  time  1.9858219623565674
Average time per epoch 1.2134631276130676 (for 500 epochs)
Epoch  279  time  2.0207834243774414
Average time per epoch 1.2175046944618224 (for 500 epochs)
Epoch  280  time  1.9362666606903076
Epoch  280  loss  0.11113992469100094 correct 50
Average time per epoch 1.2213772277832031 (for 500 epochs)
Epoch  281  time  1.9348351955413818
Average time per epoch 1.225246898174286 (for 500 epochs)
Epoch  282  time  2.0450620651245117
Average time per epoch 1.229337022304535 (for 500 epochs)
Epoch  283  time  3.0044286251068115
Average time per epoch 1.2353458795547485 (for 500 epochs)
Epoch  284  time  3.3572843074798584
Average time per epoch 1.2420604481697082 (for 500 epochs)
Epoch  285  time  3.4115326404571533
Average time per epoch 1.2488835134506227 (for 500 epochs)
Epoch  286  time  2.913551092147827
Average time per epoch 1.2547106156349181 (for 500 epochs)
Epoch  287  time  1.9437305927276611
Average time per epoch 1.2585980768203735 (for 500 epochs)
Epoch  288  time  1.966111660003662
Average time per epoch 1.262530300140381 (for 500 epochs)
Epoch  289  time  2.019636869430542
Average time per epoch 1.266569573879242 (for 500 epochs)
Epoch  290  time  1.956317663192749
Epoch  290  loss  0.0016707494259459983 correct 50
Average time per epoch 1.2704822092056274 (for 500 epochs)
Epoch  291  time  1.9641320705413818
Average time per epoch 1.2744104733467103 (for 500 epochs)
Epoch  292  time  1.9848804473876953
Average time per epoch 1.2783802342414856 (for 500 epochs)
Epoch  293  time  2.0276379585266113
Average time per epoch 1.2824355101585387 (for 500 epochs)
Epoch  294  time  1.9751298427581787
Average time per epoch 1.2863857698440553 (for 500 epochs)
Epoch  295  time  1.970062494277954
Average time per epoch 1.290325894832611 (for 500 epochs)
Epoch  296  time  2.0184271335601807
Average time per epoch 1.2943627490997314 (for 500 epochs)
Epoch  297  time  2.0034892559051514
Average time per epoch 1.2983697276115418 (for 500 epochs)
Epoch  298  time  1.9523568153381348
Average time per epoch 1.302274441242218 (for 500 epochs)
Epoch  299  time  2.0728061199188232
Average time per epoch 1.3064200534820556 (for 500 epochs)
Epoch  300  time  1.9471569061279297
Epoch  300  loss  0.13965411313657522 correct 50
Average time per epoch 1.3103143672943116 (for 500 epochs)
Epoch  301  time  1.9574956893920898
Average time per epoch 1.3142293586730958 (for 500 epochs)
Epoch  302  time  1.9296677112579346
Average time per epoch 1.3180886940956116 (for 500 epochs)
Epoch  303  time  2.038137912750244
Average time per epoch 1.322164969921112 (for 500 epochs)
Epoch  304  time  1.962186574935913
Average time per epoch 1.326089343070984 (for 500 epochs)
Epoch  305  time  1.943892240524292
Average time per epoch 1.3299771275520325 (for 500 epochs)
Epoch  306  time  2.027573585510254
Average time per epoch 1.334032274723053 (for 500 epochs)
Epoch  307  time  1.9109773635864258
Average time per epoch 1.3378542294502258 (for 500 epochs)
Epoch  308  time  1.926114797592163
Average time per epoch 1.3417064590454102 (for 500 epochs)
Epoch  309  time  1.949183464050293
Average time per epoch 1.3456048259735107 (for 500 epochs)
Epoch  310  time  2.0109357833862305
Epoch  310  loss  0.11803425362066929 correct 50
Average time per epoch 1.3496266975402833 (for 500 epochs)
Epoch  311  time  1.968550682067871
Average time per epoch 1.353563798904419 (for 500 epochs)
Epoch  312  time  1.9422225952148438
Average time per epoch 1.3574482440948485 (for 500 epochs)
Epoch  313  time  2.004023790359497
Average time per epoch 1.3614562916755677 (for 500 epochs)
Epoch  314  time  1.9319758415222168
Average time per epoch 1.365320243358612 (for 500 epochs)
Epoch  315  time  1.9231541156768799
Average time per epoch 1.3691665515899658 (for 500 epochs)
Epoch  316  time  2.1566505432128906
Average time per epoch 1.3734798526763916 (for 500 epochs)
Epoch  317  time  3.2689919471740723
Average time per epoch 1.3800178365707398 (for 500 epochs)
Epoch  318  time  3.4321019649505615
Average time per epoch 1.3868820405006408 (for 500 epochs)
Epoch  319  time  3.3897242546081543
Average time per epoch 1.393661489009857 (for 500 epochs)
Epoch  320  time  2.4961724281311035
Epoch  320  loss  0.22996523312396858 correct 50
Average time per epoch 1.3986538338661194 (for 500 epochs)
Epoch  321  time  1.982173204421997
Average time per epoch 1.4026181802749633 (for 500 epochs)
Epoch  322  time  1.9640223979949951
Average time per epoch 1.4065462250709533 (for 500 epochs)
Epoch  323  time  2.037815570831299
Average time per epoch 1.410621856212616 (for 500 epochs)
Epoch  324  time  1.9570190906524658
Average time per epoch 1.4145358943939208 (for 500 epochs)
Epoch  325  time  1.9928154945373535
Average time per epoch 1.4185215253829957 (for 500 epochs)
Epoch  326  time  1.9882194995880127
Average time per epoch 1.4224979643821716 (for 500 epochs)
Epoch  327  time  1.9637281894683838
Average time per epoch 1.4264254207611085 (for 500 epochs)
Epoch  328  time  1.937729835510254
Average time per epoch 1.4303008804321289 (for 500 epochs)
Epoch  329  time  1.9533333778381348
Average time per epoch 1.434207547187805 (for 500 epochs)
Epoch  330  time  2.0210747718811035
Epoch  330  loss  0.054546372327299385 correct 50
Average time per epoch 1.4382496967315674 (for 500 epochs)
Epoch  331  time  1.9275331497192383
Average time per epoch 1.4421047630310058 (for 500 epochs)
Epoch  332  time  1.9833509922027588
Average time per epoch 1.4460714650154114 (for 500 epochs)
Epoch  333  time  1.9994077682495117
Average time per epoch 1.4500702805519103 (for 500 epochs)
Epoch  334  time  1.9677588939666748
Average time per epoch 1.4540057983398438 (for 500 epochs)
Epoch  335  time  1.9483494758605957
Average time per epoch 1.4579024972915648 (for 500 epochs)
Epoch  336  time  1.9294743537902832
Average time per epoch 1.4617614459991455 (for 500 epochs)
Epoch  337  time  1.9915165901184082
Average time per epoch 1.4657444791793823 (for 500 epochs)
Epoch  338  time  1.978987693786621
Average time per epoch 1.4697024545669555 (for 500 epochs)
Epoch  339  time  1.9324846267700195
Average time per epoch 1.4735674238204957 (for 500 epochs)
Epoch  340  time  1.9868366718292236
Epoch  340  loss  0.0038511229609772117 correct 50
Average time per epoch 1.477541097164154 (for 500 epochs)
Epoch  341  time  1.9390997886657715
Average time per epoch 1.4814192967414856 (for 500 epochs)
Epoch  342  time  1.9578135013580322
Average time per epoch 1.4853349237442017 (for 500 epochs)
Epoch  343  time  1.9961769580841064
Average time per epoch 1.4893272776603699 (for 500 epochs)
Epoch  344  time  1.9362297058105469
Average time per epoch 1.493199737071991 (for 500 epochs)
Epoch  345  time  1.9315650463104248
Average time per epoch 1.4970628671646118 (for 500 epochs)
Epoch  346  time  1.9559156894683838
Average time per epoch 1.5009746985435486 (for 500 epochs)
Epoch  347  time  2.0162971019744873
Average time per epoch 1.5050072927474976 (for 500 epochs)
Epoch  348  time  1.927846908569336
Average time per epoch 1.5088629865646361 (for 500 epochs)
Epoch  349  time  2.2650704383850098
Average time per epoch 1.5133931274414063 (for 500 epochs)
Epoch  350  time  3.3623807430267334
Epoch  350  loss  0.05693206824047686 correct 50
Average time per epoch 1.5201178889274598 (for 500 epochs)
Epoch  351  time  3.3676860332489014
Average time per epoch 1.5268532609939576 (for 500 epochs)
Epoch  352  time  3.36871075630188
Average time per epoch 1.5335906825065613 (for 500 epochs)
Epoch  353  time  3.197765350341797
Average time per epoch 1.5399862132072448 (for 500 epochs)
Epoch  354  time  3.505784034729004
Average time per epoch 1.5469977812767028 (for 500 epochs)
Epoch  355  time  3.3739173412323
Average time per epoch 1.5537456159591674 (for 500 epochs)
Epoch  356  time  2.600987195968628
Average time per epoch 1.5589475903511048 (for 500 epochs)
Epoch  357  time  2.0457820892333984
Average time per epoch 1.5630391545295714 (for 500 epochs)
Epoch  358  time  2.00286865234375
Average time per epoch 1.567044891834259 (for 500 epochs)
Epoch  359  time  1.9608900547027588
Average time per epoch 1.5709666719436646 (for 500 epochs)
Epoch  360  time  2.0197956562042236
Epoch  360  loss  0.04346227711628855 correct 50
Average time per epoch 1.575006263256073 (for 500 epochs)
Epoch  361  time  1.9333162307739258
Average time per epoch 1.5788728957176208 (for 500 epochs)
Epoch  362  time  1.9479329586029053
Average time per epoch 1.5827687616348267 (for 500 epochs)
Epoch  363  time  1.956132411956787
Average time per epoch 1.5866810264587403 (for 500 epochs)
Epoch  364  time  2.0258262157440186
Average time per epoch 1.5907326788902283 (for 500 epochs)
Epoch  365  time  1.9321699142456055
Average time per epoch 1.5945970187187195 (for 500 epochs)
Epoch  366  time  1.9385523796081543
Average time per epoch 1.5984741234779358 (for 500 epochs)
Epoch  367  time  2.0154080390930176
Average time per epoch 1.6025049395561217 (for 500 epochs)
Epoch  368  time  1.9649016857147217
Average time per epoch 1.6064347429275512 (for 500 epochs)
Epoch  369  time  1.9495806694030762
Average time per epoch 1.6103339042663574 (for 500 epochs)
Epoch  370  time  1.9280459880828857
Epoch  370  loss  0.16039142089340055 correct 50
Average time per epoch 1.6141899962425232 (for 500 epochs)
Epoch  371  time  1.9956333637237549
Average time per epoch 1.6181812629699708 (for 500 epochs)
Epoch  372  time  1.93021559715271
Average time per epoch 1.6220416941642761 (for 500 epochs)
Epoch  373  time  1.9542829990386963
Average time per epoch 1.6259502601623534 (for 500 epochs)
Epoch  374  time  2.041214942932129
Average time per epoch 1.6300326900482178 (for 500 epochs)
Epoch  375  time  1.946514368057251
Average time per epoch 1.6339257187843323 (for 500 epochs)
Epoch  376  time  1.9596521854400635
Average time per epoch 1.6378450231552124 (for 500 epochs)
Epoch  377  time  2.019723892211914
Average time per epoch 1.6418844709396363 (for 500 epochs)
Epoch  378  time  2.8606503009796143
Average time per epoch 1.6476057715415955 (for 500 epochs)
Epoch  379  time  3.3812355995178223
Average time per epoch 1.6543682427406312 (for 500 epochs)
Epoch  380  time  3.2306101322174072
Epoch  380  loss  0.18272465868321797 correct 50
Average time per epoch 1.660829463005066 (for 500 epochs)
Epoch  381  time  3.503401041030884
Average time per epoch 1.6678362650871277 (for 500 epochs)
Epoch  382  time  3.3056960105895996
Average time per epoch 1.674447657108307 (for 500 epochs)
Epoch  383  time  3.063847064971924
Average time per epoch 1.6805753512382506 (for 500 epochs)
Epoch  384  time  1.9948971271514893
Average time per epoch 1.6845651454925537 (for 500 epochs)
Epoch  385  time  1.9876787662506104
Average time per epoch 1.6885405030250549 (for 500 epochs)
Epoch  386  time  1.9537770748138428
Average time per epoch 1.6924480571746827 (for 500 epochs)
Epoch  387  time  1.9514100551605225
Average time per epoch 1.6963508772850036 (for 500 epochs)
Epoch  388  time  2.0004637241363525
Average time per epoch 1.7003518047332764 (for 500 epochs)
Epoch  389  time  1.9467403888702393
Average time per epoch 1.7042452855110168 (for 500 epochs)
Epoch  390  time  1.9227185249328613
Epoch  390  loss  0.013493507961048373 correct 50
Average time per epoch 1.7080907225608826 (for 500 epochs)
Epoch  391  time  2.0312159061431885
Average time per epoch 1.712153154373169 (for 500 epochs)
Epoch  392  time  1.9522030353546143
Average time per epoch 1.7160575604438781 (for 500 epochs)
Epoch  393  time  1.935758352279663
Average time per epoch 1.7199290771484375 (for 500 epochs)
Epoch  394  time  2.0138378143310547
Average time per epoch 1.7239567527770996 (for 500 epochs)
Epoch  395  time  1.951920747756958
Average time per epoch 1.7278605942726135 (for 500 epochs)
Epoch  396  time  1.9531035423278809
Average time per epoch 1.7317668013572693 (for 500 epochs)
Epoch  397  time  1.9506700038909912
Average time per epoch 1.7356681413650512 (for 500 epochs)
Epoch  398  time  2.034057378768921
Average time per epoch 1.739736256122589 (for 500 epochs)
Epoch  399  time  1.938995361328125
Average time per epoch 1.7436142468452454 (for 500 epochs)
Epoch  400  time  1.9530417919158936
Epoch  400  loss  0.000514858841606241 correct 50
Average time per epoch 1.7475203304290772 (for 500 epochs)
Epoch  401  time  2.0091726779937744
Average time per epoch 1.7515386757850646 (for 500 epochs)
Epoch  402  time  1.9512689113616943
Average time per epoch 1.7554412136077882 (for 500 epochs)
Epoch  403  time  1.946237325668335
Average time per epoch 1.7593336882591248 (for 500 epochs)
Epoch  404  time  1.984632968902588
Average time per epoch 1.7633029541969298 (for 500 epochs)
Epoch  405  time  1.9294281005859375
Average time per epoch 1.7671618103981017 (for 500 epochs)
Epoch  406  time  1.93532395362854
Average time per epoch 1.771032458305359 (for 500 epochs)
Epoch  407  time  1.9457364082336426
Average time per epoch 1.7749239311218261 (for 500 epochs)
Epoch  408  time  1.999122142791748
Average time per epoch 1.7789221754074096 (for 500 epochs)
Epoch  409  time  1.94972562789917
Average time per epoch 1.782821626663208 (for 500 epochs)
Epoch  410  time  1.952322006225586
Epoch  410  loss  0.11288463527455422 correct 50
Average time per epoch 1.7867262706756593 (for 500 epochs)
Epoch  411  time  2.005479574203491
Average time per epoch 1.7907372298240662 (for 500 epochs)
Epoch  412  time  1.9348883628845215
Average time per epoch 1.7946070065498352 (for 500 epochs)
Epoch  413  time  2.1125130653381348
Average time per epoch 1.7988320326805114 (for 500 epochs)
Epoch  414  time  3.221949338912964
Average time per epoch 1.8052759313583373 (for 500 epochs)
Epoch  415  time  3.4967174530029297
Average time per epoch 1.8122693662643432 (for 500 epochs)
Epoch  416  time  3.372309684753418
Average time per epoch 1.81901398563385 (for 500 epochs)
Epoch  417  time  2.407503128051758
Average time per epoch 1.8238289918899535 (for 500 epochs)
Epoch  418  time  1.9951624870300293
Average time per epoch 1.8278193168640138 (for 500 epochs)
Epoch  419  time  1.9950358867645264
Average time per epoch 1.8318093886375428 (for 500 epochs)
Epoch  420  time  1.9701378345489502
Epoch  420  loss  0.008566531832112685 correct 50
Average time per epoch 1.8357496643066407 (for 500 epochs)
Epoch  421  time  2.0015969276428223
Average time per epoch 1.8397528581619262 (for 500 epochs)
Epoch  422  time  1.9538171291351318
Average time per epoch 1.8436604924201965 (for 500 epochs)
Epoch  423  time  1.9367132186889648
Average time per epoch 1.8475339188575746 (for 500 epochs)
Epoch  424  time  1.9177429676055908
Average time per epoch 1.8513694047927856 (for 500 epochs)
Epoch  425  time  1.9892899990081787
Average time per epoch 1.855347984790802 (for 500 epochs)
Epoch  426  time  1.9264647960662842
Average time per epoch 1.8592009143829347 (for 500 epochs)
Epoch  427  time  1.928621530532837
Average time per epoch 1.8630581574440002 (for 500 epochs)
Epoch  428  time  2.008333683013916
Average time per epoch 1.8670748248100282 (for 500 epochs)
Epoch  429  time  1.9649274349212646
Average time per epoch 1.8710046796798705 (for 500 epochs)
Epoch  430  time  1.9720799922943115
Epoch  430  loss  0.08774895512966106 correct 50
Average time per epoch 1.8749488396644591 (for 500 epochs)
Epoch  431  time  1.9380519390106201
Average time per epoch 1.8788249435424804 (for 500 epochs)
Epoch  432  time  2.007384777069092
Average time per epoch 1.8828397130966186 (for 500 epochs)
Epoch  433  time  1.9585919380187988
Average time per epoch 1.8867568969726562 (for 500 epochs)
Epoch  434  time  1.956928014755249
Average time per epoch 1.8906707530021667 (for 500 epochs)
Epoch  435  time  2.0381011962890625
Average time per epoch 1.894746955394745 (for 500 epochs)
Epoch  436  time  1.9682672023773193
Average time per epoch 1.8986834897994995 (for 500 epochs)
Epoch  437  time  2.002300977706909
Average time per epoch 1.9026880917549134 (for 500 epochs)
Epoch  438  time  2.0575244426727295
Average time per epoch 1.9068031406402588 (for 500 epochs)
Epoch  439  time  1.9267916679382324
Average time per epoch 1.9106567239761352 (for 500 epochs)
Epoch  440  time  1.9968187808990479
Epoch  440  loss  0.03261684943953743 correct 50
Average time per epoch 1.9146503615379333 (for 500 epochs)
Epoch  441  time  1.943552017211914
Average time per epoch 1.9185374655723573 (for 500 epochs)
Epoch  442  time  2.0177133083343506
Average time per epoch 1.922572892189026 (for 500 epochs)
Epoch  443  time  1.954176425933838
Average time per epoch 1.9264812450408935 (for 500 epochs)
Epoch  444  time  1.9230899810791016
Average time per epoch 1.9303274250030518 (for 500 epochs)
Epoch  445  time  2.043903112411499
Average time per epoch 1.9344152312278748 (for 500 epochs)
Epoch  446  time  1.9535155296325684
Average time per epoch 1.9383222622871399 (for 500 epochs)
Epoch  447  time  1.9183640480041504
Average time per epoch 1.9421589903831482 (for 500 epochs)
Epoch  448  time  3.0647897720336914
Average time per epoch 1.9482885699272157 (for 500 epochs)
Epoch  449  time  3.339519739151001
Average time per epoch 1.9549676094055175 (for 500 epochs)
Epoch  450  time  3.357240676879883
Epoch  450  loss  0.0440528380030283 correct 50
Average time per epoch 1.9616820907592774 (for 500 epochs)
Epoch  451  time  2.764801263809204
Average time per epoch 1.9672116932868957 (for 500 epochs)
Epoch  452  time  1.999842882156372
Average time per epoch 1.9712113790512085 (for 500 epochs)
Epoch  453  time  1.921168327331543
Average time per epoch 1.9750537157058716 (for 500 epochs)
Epoch  454  time  1.9185872077941895
Average time per epoch 1.97889089012146 (for 500 epochs)
Epoch  455  time  1.9885129928588867
Average time per epoch 1.9828679161071778 (for 500 epochs)
Epoch  456  time  1.9473435878753662
Average time per epoch 1.9867626032829284 (for 500 epochs)
Epoch  457  time  1.9679069519042969
Average time per epoch 1.990698417186737 (for 500 epochs)
Epoch  458  time  1.9668657779693604
Average time per epoch 1.9946321487426757 (for 500 epochs)
Epoch  459  time  2.045260190963745
Average time per epoch 1.9987226691246032 (for 500 epochs)
Epoch  460  time  1.95713210105896
Epoch  460  loss  0.14572131621118078 correct 50
Average time per epoch 2.002636933326721 (for 500 epochs)
Epoch  461  time  1.9222068786621094
Average time per epoch 2.0064813470840455 (for 500 epochs)
Epoch  462  time  2.025614023208618
Average time per epoch 2.010532575130463 (for 500 epochs)
Epoch  463  time  1.925574541091919
Average time per epoch 2.0143837242126463 (for 500 epochs)
Epoch  464  time  1.936497688293457
Average time per epoch 2.0182567195892336 (for 500 epochs)
Epoch  465  time  1.9791574478149414
Average time per epoch 2.022215034484863 (for 500 epochs)
Epoch  466  time  2.0018935203552246
Average time per epoch 2.026218821525574 (for 500 epochs)
Epoch  467  time  1.9426662921905518
Average time per epoch 2.0301041541099547 (for 500 epochs)
Epoch  468  time  1.9354979991912842
Average time per epoch 2.033975150108337 (for 500 epochs)
Epoch  469  time  1.9820261001586914
Average time per epoch 2.037939202308655 (for 500 epochs)
Epoch  470  time  1.9345409870147705
Epoch  470  loss  0.23514882250870053 correct 50
Average time per epoch 2.041808284282684 (for 500 epochs)
Epoch  471  time  1.9274425506591797
Average time per epoch 2.0456631693840026 (for 500 epochs)
Epoch  472  time  1.99300217628479
Average time per epoch 2.0496491737365723 (for 500 epochs)
Epoch  473  time  1.9401893615722656
Average time per epoch 2.053529552459717 (for 500 epochs)
Epoch  474  time  1.9728219509124756
Average time per epoch 2.057475196361542 (for 500 epochs)
Epoch  475  time  1.9398455619812012
Average time per epoch 2.0613548874855043 (for 500 epochs)
Epoch  476  time  1.9715418815612793
Average time per epoch 2.0652979712486266 (for 500 epochs)
Epoch  477  time  1.9847180843353271
Average time per epoch 2.069267407417297 (for 500 epochs)
Epoch  478  time  1.9226605892181396
Average time per epoch 2.0731127285957336 (for 500 epochs)
Epoch  479  time  2.0014028549194336
Average time per epoch 2.0771155343055727 (for 500 epochs)
Epoch  480  time  1.9204084873199463
Epoch  480  loss  0.054076380585379204 correct 50
Average time per epoch 2.0809563512802125 (for 500 epochs)
Epoch  481  time  1.9468474388122559
Average time per epoch 2.084850046157837 (for 500 epochs)
Epoch  482  time  3.0021824836730957
Average time per epoch 2.090854411125183 (for 500 epochs)
Epoch  483  time  3.319622755050659
Average time per epoch 2.0974936566352844 (for 500 epochs)
Epoch  484  time  3.3596227169036865
Average time per epoch 2.104212902069092 (for 500 epochs)
Epoch  485  time  2.868844747543335
Average time per epoch 2.1099505915641785 (for 500 epochs)
Epoch  486  time  1.9824330806732178
Average time per epoch 2.113915457725525 (for 500 epochs)
Epoch  487  time  1.9818718433380127
Average time per epoch 2.117879201412201 (for 500 epochs)
Epoch  488  time  1.936633825302124
Average time per epoch 2.1217524690628053 (for 500 epochs)
Epoch  489  time  2.011603593826294
Average time per epoch 2.125775676250458 (for 500 epochs)
Epoch  490  time  1.9461143016815186
Epoch  490  loss  0.18482039738706818 correct 50
Average time per epoch 2.1296679048538207 (for 500 epochs)
Epoch  491  time  1.9226927757263184
Average time per epoch 2.1335132904052734 (for 500 epochs)
Epoch  492  time  1.9541041851043701
Average time per epoch 2.137421498775482 (for 500 epochs)
Epoch  493  time  3.271635055541992
Average time per epoch 2.143964768886566 (for 500 epochs)
Epoch  494  time  3.3551151752471924
Average time per epoch 2.1506749992370606 (for 500 epochs)
Epoch  495  time  3.4045112133026123
Average time per epoch 2.1574840216636657 (for 500 epochs)
Epoch  496  time  2.6491541862487793
Average time per epoch 2.1627823300361633 (for 500 epochs)
Epoch  497  time  1.9302918910980225
Average time per epoch 2.166642913818359 (for 500 epochs)
Epoch  498  time  1.957627296447754
Average time per epoch 2.170558168411255 (for 500 epochs)
Epoch  499  time  1.984039068222046
Average time per epoch 2.174526246547699 (for 500 epochs)

```
# GPU Split Dataset:
```
Epoch  0  loss  8.022012595375895 correct 34
Average time per epoch 0.010082674026489259 (for 500 epochs)
Epoch  1  time  2.0514180660247803
Average time per epoch 0.014185510158538818 (for 500 epochs)
Epoch  2  time  2.0097172260284424
Average time per epoch 0.018204944610595703 (for 500 epochs)
Epoch  3  time  1.9975261688232422
Average time per epoch 0.022199996948242186 (for 500 epochs)
Epoch  4  time  1.9747538566589355
Average time per epoch 0.026149504661560057 (for 500 epochs)
Epoch  5  time  2.0883188247680664
Average time per epoch 0.03032614231109619 (for 500 epochs)
Epoch  6  time  1.967254638671875
Average time per epoch 0.03426065158843994 (for 500 epochs)
Epoch  7  time  1.9649112224578857
Average time per epoch 0.038190474033355716 (for 500 epochs)
Epoch  8  time  2.067441940307617
Average time per epoch 0.04232535791397095 (for 500 epochs)
Epoch  9  time  1.9940242767333984
Average time per epoch 0.04631340646743774 (for 500 epochs)
Epoch  10  time  1.9566318988800049
Epoch  10  loss  4.686172522547588 correct 43
Average time per epoch 0.050226670265197754 (for 500 epochs)
Epoch  11  time  2.078784465789795
Average time per epoch 0.054384239196777345 (for 500 epochs)
Epoch  12  time  1.9758787155151367
Average time per epoch 0.05833599662780762 (for 500 epochs)
Epoch  13  time  1.9775266647338867
Average time per epoch 0.06229104995727539 (for 500 epochs)
Epoch  14  time  1.9783506393432617
Average time per epoch 0.06624775123596191 (for 500 epochs)
Epoch  15  time  2.1134192943573
Average time per epoch 0.07047458982467651 (for 500 epochs)
Epoch  16  time  1.9790639877319336
Average time per epoch 0.07443271780014038 (for 500 epochs)
Epoch  17  time  1.9810433387756348
Average time per epoch 0.07839480447769165 (for 500 epochs)
Epoch  18  time  2.1049797534942627
Average time per epoch 0.08260476398468018 (for 500 epochs)
Epoch  19  time  2.01731276512146
Average time per epoch 0.0866393895149231 (for 500 epochs)
Epoch  20  time  2.0134596824645996
Epoch  20  loss  4.500284596597121 correct 46
Average time per epoch 0.09066630887985229 (for 500 epochs)
Epoch  21  time  2.0123131275177
Average time per epoch 0.0946909351348877 (for 500 epochs)
Epoch  22  time  2.071540355682373
Average time per epoch 0.09883401584625244 (for 500 epochs)
Epoch  23  time  1.979440689086914
Average time per epoch 0.10279289722442626 (for 500 epochs)
Epoch  24  time  2.0148162841796875
Average time per epoch 0.10682252979278564 (for 500 epochs)
Epoch  25  time  3.2026777267456055
Average time per epoch 0.11322788524627686 (for 500 epochs)
Epoch  26  time  3.4534101486206055
Average time per epoch 0.12013470554351807 (for 500 epochs)
Epoch  27  time  3.4281165599823
Average time per epoch 0.12699093866348266 (for 500 epochs)
Epoch  28  time  3.404859781265259
Average time per epoch 0.1338006582260132 (for 500 epochs)
Epoch  29  time  3.538341760635376
Average time per epoch 0.14087734174728395 (for 500 epochs)
Epoch  30  time  3.4624884128570557
Epoch  30  loss  3.627796166969849 correct 45
Average time per epoch 0.14780231857299805 (for 500 epochs)
Epoch  31  time  2.3828518390655518
Average time per epoch 0.15256802225112914 (for 500 epochs)
Epoch  32  time  2.0307343006134033
Average time per epoch 0.15662949085235595 (for 500 epochs)
Epoch  33  time  1.9800848960876465
Average time per epoch 0.16058966064453126 (for 500 epochs)
Epoch  34  time  1.976954698562622
Average time per epoch 0.16454357004165648 (for 500 epochs)
Epoch  35  time  2.06611704826355
Average time per epoch 0.1686758041381836 (for 500 epochs)
Epoch  36  time  2.0053603649139404
Average time per epoch 0.17268652486801148 (for 500 epochs)
Epoch  37  time  2.0070998668670654
Average time per epoch 0.1767007246017456 (for 500 epochs)
Epoch  38  time  1.9572288990020752
Average time per epoch 0.18061518239974975 (for 500 epochs)
Epoch  39  time  2.044121026992798
Average time per epoch 0.18470342445373536 (for 500 epochs)
Epoch  40  time  1.9948499202728271
Epoch  40  loss  2.9401795351916133 correct 48
Average time per epoch 0.188693124294281 (for 500 epochs)
Epoch  41  time  1.9754905700683594
Average time per epoch 0.19264410543441773 (for 500 epochs)
Epoch  42  time  2.093468427658081
Average time per epoch 0.1968310422897339 (for 500 epochs)
Epoch  43  time  2.0222814083099365
Average time per epoch 0.20087560510635377 (for 500 epochs)
Epoch  44  time  2.006121873855591
Average time per epoch 0.20488784885406494 (for 500 epochs)
Epoch  45  time  2.049809694290161
Average time per epoch 0.20898746824264527 (for 500 epochs)
Epoch  46  time  2.0039360523223877
Average time per epoch 0.21299534034729004 (for 500 epochs)
Epoch  47  time  1.9660041332244873
Average time per epoch 0.21692734861373902 (for 500 epochs)
Epoch  48  time  2.009410858154297
Average time per epoch 0.2209461703300476 (for 500 epochs)
Epoch  49  time  2.064993381500244
Average time per epoch 0.2250761570930481 (for 500 epochs)
Epoch  50  time  1.978722333908081
Epoch  50  loss  2.8181016114058446 correct 48
Average time per epoch 0.22903360176086426 (for 500 epochs)
Epoch  51  time  2.0407707691192627
Average time per epoch 0.2331151432991028 (for 500 epochs)
Epoch  52  time  2.040393590927124
Average time per epoch 0.23719593048095702 (for 500 epochs)
Epoch  53  time  2.0139482021331787
Average time per epoch 0.2412238268852234 (for 500 epochs)
Epoch  54  time  1.9715986251831055
Average time per epoch 0.2451670241355896 (for 500 epochs)
Epoch  55  time  2.673753499984741
Average time per epoch 0.2505145311355591 (for 500 epochs)
Epoch  56  time  3.487182855606079
Average time per epoch 0.25748889684677123 (for 500 epochs)
Epoch  57  time  3.5445425510406494
Average time per epoch 0.2645779819488525 (for 500 epochs)
Epoch  58  time  3.6249210834503174
Average time per epoch 0.2718278241157532 (for 500 epochs)
Epoch  59  time  2.747673273086548
Average time per epoch 0.27732317066192624 (for 500 epochs)
Epoch  60  time  1.9791114330291748
Epoch  60  loss  3.200107147056752 correct 43
Average time per epoch 0.28128139352798465 (for 500 epochs)
Epoch  61  time  1.9783978462219238
Average time per epoch 0.2852381892204285 (for 500 epochs)
Epoch  62  time  2.0418596267700195
Average time per epoch 0.2893219084739685 (for 500 epochs)
Epoch  63  time  2.00541615486145
Average time per epoch 0.2933327407836914 (for 500 epochs)
Epoch  64  time  2.023491382598877
Average time per epoch 0.29737972354888914 (for 500 epochs)
Epoch  65  time  1.992814064025879
Average time per epoch 0.3013653516769409 (for 500 epochs)
Epoch  66  time  2.0546510219573975
Average time per epoch 0.3054746537208557 (for 500 epochs)
Epoch  67  time  2.0079386234283447
Average time per epoch 0.3094905309677124 (for 500 epochs)
Epoch  68  time  1.968371868133545
Average time per epoch 0.3134272747039795 (for 500 epochs)
Epoch  69  time  2.1090610027313232
Average time per epoch 0.31764539670944214 (for 500 epochs)
Epoch  70  time  2.000704288482666
Epoch  70  loss  1.3066418312078638 correct 50
Average time per epoch 0.3216468052864075 (for 500 epochs)
Epoch  71  time  1.9571192264556885
Average time per epoch 0.3255610437393188 (for 500 epochs)
Epoch  72  time  2.045480489730835
Average time per epoch 0.32965200471878053 (for 500 epochs)
Epoch  73  time  2.0050759315490723
Average time per epoch 0.33366215658187864 (for 500 epochs)
Epoch  74  time  1.9704291820526123
Average time per epoch 0.3376030149459839 (for 500 epochs)
Epoch  75  time  1.9679176807403564
Average time per epoch 0.3415388503074646 (for 500 epochs)
Epoch  76  time  2.0324959754943848
Average time per epoch 0.34560384225845336 (for 500 epochs)
Epoch  77  time  1.970644235610962
Average time per epoch 0.3495451307296753 (for 500 epochs)
Epoch  78  time  1.9758415222167969
Average time per epoch 0.3534968137741089 (for 500 epochs)
Epoch  79  time  2.028287172317505
Average time per epoch 0.3575533881187439 (for 500 epochs)
Epoch  80  time  1.9479055404663086
Epoch  80  loss  1.3438933853373414 correct 50
Average time per epoch 0.3614491991996765 (for 500 epochs)
Epoch  81  time  1.9479923248291016
Average time per epoch 0.36534518384933473 (for 500 epochs)
Epoch  82  time  1.946800708770752
Average time per epoch 0.3692387852668762 (for 500 epochs)
Epoch  83  time  2.067741870880127
Average time per epoch 0.37337426900863646 (for 500 epochs)
Epoch  84  time  1.9543156623840332
Average time per epoch 0.37728290033340456 (for 500 epochs)
Epoch  85  time  1.9675898551940918
Average time per epoch 0.3812180800437927 (for 500 epochs)
Epoch  86  time  2.035846710205078
Average time per epoch 0.3852897734642029 (for 500 epochs)
Epoch  87  time  1.9634599685668945
Average time per epoch 0.3892166934013367 (for 500 epochs)
Epoch  88  time  1.9969727993011475
Average time per epoch 0.39321063899993897 (for 500 epochs)
Epoch  89  time  2.0357248783111572
Average time per epoch 0.3972820887565613 (for 500 epochs)
Epoch  90  time  2.450115203857422
Epoch  90  loss  0.49415399184400666 correct 50
Average time per epoch 0.4021823191642761 (for 500 epochs)
Epoch  91  time  3.3832345008850098
Average time per epoch 0.40894878816604613 (for 500 epochs)
Epoch  92  time  3.455167531967163
Average time per epoch 0.4158591232299805 (for 500 epochs)
Epoch  93  time  3.4205172061920166
Average time per epoch 0.4227001576423645 (for 500 epochs)
Epoch  94  time  1.9778125286102295
Average time per epoch 0.426655782699585 (for 500 epochs)
Epoch  95  time  1.9568395614624023
Average time per epoch 0.43056946182250977 (for 500 epochs)
Epoch  96  time  2.029249668121338
Average time per epoch 0.43462796115875246 (for 500 epochs)
Epoch  97  time  1.9712765216827393
Average time per epoch 0.4385705142021179 (for 500 epochs)
Epoch  98  time  1.9796676635742188
Average time per epoch 0.4425298495292664 (for 500 epochs)
Epoch  99  time  1.9641392230987549
Average time per epoch 0.44645812797546386 (for 500 epochs)
Epoch  100  time  2.0073912143707275
Epoch  100  loss  0.398836301959542 correct 48
Average time per epoch 0.45047291040420534 (for 500 epochs)
Epoch  101  time  2.015761375427246
Average time per epoch 0.45450443315505984 (for 500 epochs)
Epoch  102  time  1.99222993850708
Average time per epoch 0.45848889303207396 (for 500 epochs)
Epoch  103  time  2.0192277431488037
Average time per epoch 0.4625273485183716 (for 500 epochs)
Epoch  104  time  1.9595913887023926
Average time per epoch 0.46644653129577635 (for 500 epochs)
Epoch  105  time  1.9426500797271729
Average time per epoch 0.4703318314552307 (for 500 epochs)
Epoch  106  time  2.0096495151519775
Average time per epoch 0.4743511304855347 (for 500 epochs)
Epoch  107  time  1.9788808822631836
Average time per epoch 0.478308892250061 (for 500 epochs)
Epoch  108  time  1.9517180919647217
Average time per epoch 0.4822123284339905 (for 500 epochs)
Epoch  109  time  1.9405295848846436
Average time per epoch 0.48609338760375975 (for 500 epochs)
Epoch  110  time  2.0021839141845703
Epoch  110  loss  0.4587088518962382 correct 50
Average time per epoch 0.4900977554321289 (for 500 epochs)
Epoch  111  time  1.9316506385803223
Average time per epoch 0.4939610567092895 (for 500 epochs)
Epoch  112  time  1.9508261680603027
Average time per epoch 0.49786270904541013 (for 500 epochs)
Epoch  113  time  2.0334112644195557
Average time per epoch 0.5019295315742492 (for 500 epochs)
Epoch  114  time  1.9500961303710938
Average time per epoch 0.5058297238349915 (for 500 epochs)
Epoch  115  time  1.9487025737762451
Average time per epoch 0.5097271289825439 (for 500 epochs)
Epoch  116  time  2.041978359222412
Average time per epoch 0.5138110857009888 (for 500 epochs)
Epoch  117  time  1.9542033672332764
Average time per epoch 0.5177194924354553 (for 500 epochs)
Epoch  118  time  1.9300312995910645
Average time per epoch 0.5215795550346375 (for 500 epochs)
Epoch  119  time  1.9254837036132812
Average time per epoch 0.5254305224418641 (for 500 epochs)
Epoch  120  time  1.991373062133789
Epoch  120  loss  0.7962919013660107 correct 50
Average time per epoch 0.5294132685661316 (for 500 epochs)
Epoch  121  time  1.9208276271820068
Average time per epoch 0.5332549238204956 (for 500 epochs)
Epoch  122  time  1.931361198425293
Average time per epoch 0.5371176462173461 (for 500 epochs)
Epoch  123  time  2.008493185043335
Average time per epoch 0.5411346325874329 (for 500 epochs)
Epoch  124  time  3.004960298538208
Average time per epoch 0.5471445531845093 (for 500 epochs)
Epoch  125  time  3.359452247619629
Average time per epoch 0.5538634576797485 (for 500 epochs)
Epoch  126  time  3.4505739212036133
Average time per epoch 0.5607646055221558 (for 500 epochs)
Epoch  127  time  2.8566362857818604
Average time per epoch 0.5664778780937195 (for 500 epochs)
Epoch  128  time  1.912224531173706
Average time per epoch 0.5703023271560669 (for 500 epochs)
Epoch  129  time  1.9457266330718994
Average time per epoch 0.5741937804222107 (for 500 epochs)
Epoch  130  time  2.0197994709014893
Epoch  130  loss  0.3993343858419637 correct 50
Average time per epoch 0.5782333793640136 (for 500 epochs)
Epoch  131  time  1.9478135108947754
Average time per epoch 0.5821290063858032 (for 500 epochs)
Epoch  132  time  1.959329605102539
Average time per epoch 0.5860476655960083 (for 500 epochs)
Epoch  133  time  2.0077006816864014
Average time per epoch 0.5900630669593812 (for 500 epochs)
Epoch  134  time  1.960864543914795
Average time per epoch 0.5939847960472107 (for 500 epochs)
Epoch  135  time  1.9910612106323242
Average time per epoch 0.5979669184684754 (for 500 epochs)
Epoch  136  time  1.9406957626342773
Average time per epoch 0.601848309993744 (for 500 epochs)
Epoch  137  time  2.0002331733703613
Average time per epoch 0.6058487763404846 (for 500 epochs)
Epoch  138  time  1.9823145866394043
Average time per epoch 0.6098134055137634 (for 500 epochs)
Epoch  139  time  1.9697329998016357
Average time per epoch 0.6137528715133667 (for 500 epochs)
Epoch  140  time  1.9997353553771973
Epoch  140  loss  0.8055235523101912 correct 50
Average time per epoch 0.6177523422241211 (for 500 epochs)
Epoch  141  time  1.9621984958648682
Average time per epoch 0.6216767392158509 (for 500 epochs)
Epoch  142  time  1.9466583728790283
Average time per epoch 0.6255700559616089 (for 500 epochs)
Epoch  143  time  1.9561352729797363
Average time per epoch 0.6294823265075684 (for 500 epochs)
Epoch  144  time  2.0097908973693848
Average time per epoch 0.6335019083023071 (for 500 epochs)
Epoch  145  time  1.9745595455169678
Average time per epoch 0.637451027393341 (for 500 epochs)
Epoch  146  time  1.9331471920013428
Average time per epoch 0.6413173217773438 (for 500 epochs)
Epoch  147  time  1.986534833908081
Average time per epoch 0.6452903914451599 (for 500 epochs)
Epoch  148  time  1.9112167358398438
Average time per epoch 0.6491128249168396 (for 500 epochs)
Epoch  149  time  1.9346070289611816
Average time per epoch 0.6529820389747619 (for 500 epochs)
Epoch  150  time  2.0114002227783203
Epoch  150  loss  0.4733628596404219 correct 50
Average time per epoch 0.6570048394203186 (for 500 epochs)
Epoch  151  time  1.9415643215179443
Average time per epoch 0.6608879680633545 (for 500 epochs)
Epoch  152  time  1.96671462059021
Average time per epoch 0.6648213973045349 (for 500 epochs)
Epoch  153  time  1.9286296367645264
Average time per epoch 0.6686786565780639 (for 500 epochs)
Epoch  154  time  1.990633487701416
Average time per epoch 0.6726599235534668 (for 500 epochs)
Epoch  155  time  1.9731945991516113
Average time per epoch 0.67660631275177 (for 500 epochs)
Epoch  156  time  1.9712131023406982
Average time per epoch 0.6805487389564514 (for 500 epochs)
Epoch  157  time  2.0516197681427
Average time per epoch 0.6846519784927368 (for 500 epochs)
Epoch  158  time  3.1838266849517822
Average time per epoch 0.6910196318626404 (for 500 epochs)
Epoch  159  time  3.424457311630249
Average time per epoch 0.6978685464859009 (for 500 epochs)
Epoch  160  time  3.48551082611084
Epoch  160  loss  0.227291237922734 correct 49
Average time per epoch 0.7048395681381225 (for 500 epochs)
Epoch  161  time  2.494858503341675
Average time per epoch 0.7098292851448059 (for 500 epochs)
Epoch  162  time  1.93878173828125
Average time per epoch 0.7137068486213685 (for 500 epochs)
Epoch  163  time  2.014612913131714
Average time per epoch 0.7177360744476319 (for 500 epochs)
Epoch  164  time  2.0148580074310303
Average time per epoch 0.7217657904624939 (for 500 epochs)
Epoch  165  time  2.8809127807617188
Average time per epoch 0.7275276160240174 (for 500 epochs)
Epoch  166  time  3.4177284240722656
Average time per epoch 0.7343630728721618 (for 500 epochs)
Epoch  167  time  3.50405216217041
Average time per epoch 0.7413711771965027 (for 500 epochs)
Epoch  168  time  2.8575832843780518
Average time per epoch 0.7470863437652588 (for 500 epochs)
Epoch  169  time  1.9631750583648682
Average time per epoch 0.7510126938819885 (for 500 epochs)
Epoch  170  time  1.9700663089752197
Epoch  170  loss  0.38622131866575216 correct 50
Average time per epoch 0.754952826499939 (for 500 epochs)
Epoch  171  time  2.0329525470733643
Average time per epoch 0.7590187315940857 (for 500 epochs)
Epoch  172  time  1.9255785942077637
Average time per epoch 0.7628698887825012 (for 500 epochs)
Epoch  173  time  1.9636600017547607
Average time per epoch 0.7667972087860108 (for 500 epochs)
Epoch  174  time  2.02169132232666
Average time per epoch 0.7708405914306641 (for 500 epochs)
Epoch  175  time  1.9384543895721436
Average time per epoch 0.7747175002098083 (for 500 epochs)
Epoch  176  time  1.942807674407959
Average time per epoch 0.7786031155586243 (for 500 epochs)
Epoch  177  time  1.9941225051879883
Average time per epoch 0.7825913605690002 (for 500 epochs)
Epoch  178  time  1.938347339630127
Average time per epoch 0.7864680552482605 (for 500 epochs)
Epoch  179  time  1.953653335571289
Average time per epoch 0.790375361919403 (for 500 epochs)
Epoch  180  time  1.937671184539795
Epoch  180  loss  0.3134555982689867 correct 50
Average time per epoch 0.7942507042884827 (for 500 epochs)
Epoch  181  time  2.0418107509613037
Average time per epoch 0.7983343257904053 (for 500 epochs)
Epoch  182  time  1.9915432929992676
Average time per epoch 0.8023174123764039 (for 500 epochs)
Epoch  183  time  1.952707290649414
Average time per epoch 0.8062228269577026 (for 500 epochs)
Epoch  184  time  2.0147578716278076
Average time per epoch 0.8102523427009583 (for 500 epochs)
Epoch  185  time  1.9443504810333252
Average time per epoch 0.8141410436630249 (for 500 epochs)
Epoch  186  time  1.9846267700195312
Average time per epoch 0.8181102972030639 (for 500 epochs)
Epoch  187  time  2.0285239219665527
Average time per epoch 0.8221673450469971 (for 500 epochs)
Epoch  188  time  1.9961154460906982
Average time per epoch 0.8261595759391784 (for 500 epochs)
Epoch  189  time  3.124391555786133
Average time per epoch 0.8324083590507507 (for 500 epochs)
Epoch  190  time  3.3944127559661865
Epoch  190  loss  0.18995282063880384 correct 50
Average time per epoch 0.8391971845626831 (for 500 epochs)
Epoch  191  time  3.3681788444519043
Average time per epoch 0.8459335422515869 (for 500 epochs)
Epoch  192  time  3.379136562347412
Average time per epoch 0.8526918153762817 (for 500 epochs)
Epoch  193  time  3.370619535446167
Average time per epoch 0.8594330544471741 (for 500 epochs)
Epoch  194  time  3.4444022178649902
Average time per epoch 0.866321858882904 (for 500 epochs)
Epoch  195  time  2.0276668071746826
Average time per epoch 0.8703771924972534 (for 500 epochs)
Epoch  196  time  1.9189743995666504
Average time per epoch 0.8742151412963867 (for 500 epochs)
Epoch  197  time  2.003157138824463
Average time per epoch 0.8782214555740356 (for 500 epochs)
Epoch  198  time  2.010666847229004
Average time per epoch 0.8822427892684936 (for 500 epochs)
Epoch  199  time  1.9390108585357666
Average time per epoch 0.8861208109855652 (for 500 epochs)
Epoch  200  time  1.9502291679382324
Epoch  200  loss  0.13007555568467266 correct 50
Average time per epoch 0.8900212693214417 (for 500 epochs)
Epoch  201  time  2.0136935710906982
Average time per epoch 0.8940486564636231 (for 500 epochs)
Epoch  202  time  1.9463260173797607
Average time per epoch 0.8979413084983826 (for 500 epochs)
Epoch  203  time  1.9730885028839111
Average time per epoch 0.9018874855041504 (for 500 epochs)
Epoch  204  time  2.032238245010376
Average time per epoch 0.9059519619941712 (for 500 epochs)
Epoch  205  time  1.9531550407409668
Average time per epoch 0.9098582720756531 (for 500 epochs)
Epoch  206  time  1.9659786224365234
Average time per epoch 0.9137902293205261 (for 500 epochs)
Epoch  207  time  1.9545140266418457
Average time per epoch 0.9176992573738099 (for 500 epochs)
Epoch  208  time  2.1201746463775635
Average time per epoch 0.9219396066665649 (for 500 epochs)
Epoch  209  time  2.018327474594116
Average time per epoch 0.9259762616157532 (for 500 epochs)
Epoch  210  time  1.9562163352966309
Epoch  210  loss  0.1972819779429671 correct 50
Average time per epoch 0.9298886942863465 (for 500 epochs)
Epoch  211  time  2.01594877243042
Average time per epoch 0.9339205918312072 (for 500 epochs)
Epoch  212  time  1.9919180870056152
Average time per epoch 0.9379044280052186 (for 500 epochs)
Epoch  213  time  1.9739882946014404
Average time per epoch 0.9418524045944214 (for 500 epochs)
Epoch  214  time  1.955329179763794
Average time per epoch 0.945763062953949 (for 500 epochs)
Epoch  215  time  2.012453556060791
Average time per epoch 0.9497879700660705 (for 500 epochs)
Epoch  216  time  1.9280989170074463
Average time per epoch 0.9536441679000854 (for 500 epochs)
Epoch  217  time  1.957672357559204
Average time per epoch 0.9575595126152039 (for 500 epochs)
Epoch  218  time  2.0183632373809814
Average time per epoch 0.9615962390899658 (for 500 epochs)
Epoch  219  time  1.929809331893921
Average time per epoch 0.9654558577537536 (for 500 epochs)
Epoch  220  time  2.241727828979492
Epoch  220  loss  0.2655082980758451 correct 50
Average time per epoch 0.9699393134117127 (for 500 epochs)
Epoch  221  time  3.388554573059082
Average time per epoch 0.9767164225578309 (for 500 epochs)
Epoch  222  time  3.391641855239868
Average time per epoch 0.9834997062683105 (for 500 epochs)
Epoch  223  time  3.3750553131103516
Average time per epoch 0.9902498168945313 (for 500 epochs)
Epoch  224  time  2.2275021076202393
Average time per epoch 0.9947048211097718 (for 500 epochs)
Epoch  225  time  2.0143213272094727
Average time per epoch 0.9987334637641907 (for 500 epochs)
Epoch  226  time  1.9516246318817139
Average time per epoch 1.002636713027954 (for 500 epochs)
Epoch  227  time  1.9525058269500732
Average time per epoch 1.0065417246818542 (for 500 epochs)
Epoch  228  time  2.014781951904297
Average time per epoch 1.0105712885856628 (for 500 epochs)
Epoch  229  time  1.9301819801330566
Average time per epoch 1.014431652545929 (for 500 epochs)
Epoch  230  time  1.938521385192871
Epoch  230  loss  0.12354376448060791 correct 50
Average time per epoch 1.0183086953163147 (for 500 epochs)
Epoch  231  time  1.9519760608673096
Average time per epoch 1.0222126474380493 (for 500 epochs)
Epoch  232  time  2.021683931350708
Average time per epoch 1.0262560153007507 (for 500 epochs)
Epoch  233  time  1.9431989192962646
Average time per epoch 1.0301424131393433 (for 500 epochs)
Epoch  234  time  1.9669501781463623
Average time per epoch 1.034076313495636 (for 500 epochs)
Epoch  235  time  2.035226583480835
Average time per epoch 1.0381467666625976 (for 500 epochs)
Epoch  236  time  1.9873809814453125
Average time per epoch 1.0421215286254883 (for 500 epochs)
Epoch  237  time  1.9234626293182373
Average time per epoch 1.0459684538841247 (for 500 epochs)
Epoch  238  time  1.9530727863311768
Average time per epoch 1.049874599456787 (for 500 epochs)
Epoch  239  time  1.9956440925598145
Average time per epoch 1.0538658876419067 (for 500 epochs)
Epoch  240  time  1.9319279193878174
Epoch  240  loss  0.28513836709478346 correct 50
Average time per epoch 1.0577297434806823 (for 500 epochs)
Epoch  241  time  1.9315125942230225
Average time per epoch 1.0615927686691284 (for 500 epochs)
Epoch  242  time  2.046342134475708
Average time per epoch 1.06568545293808 (for 500 epochs)
Epoch  243  time  1.9266798496246338
Average time per epoch 1.069538812637329 (for 500 epochs)
Epoch  244  time  1.9387776851654053
Average time per epoch 1.0734163680076598 (for 500 epochs)
Epoch  245  time  2.0540661811828613
Average time per epoch 1.0775245003700256 (for 500 epochs)
Epoch  246  time  1.9249587059020996
Average time per epoch 1.0813744177818299 (for 500 epochs)
Epoch  247  time  1.931861162185669
Average time per epoch 1.0852381401062012 (for 500 epochs)
Epoch  248  time  1.9111838340759277
Average time per epoch 1.089060507774353 (for 500 epochs)
Epoch  249  time  1.9909825325012207
Average time per epoch 1.0930424728393555 (for 500 epochs)
Epoch  250  time  1.920058012008667
Epoch  250  loss  0.28033460919530834 correct 50
Average time per epoch 1.0968825888633729 (for 500 epochs)
Epoch  251  time  1.9730017185211182
Average time per epoch 1.100828592300415 (for 500 epochs)
Epoch  252  time  2.0020246505737305
Average time per epoch 1.1048326416015626 (for 500 epochs)
Epoch  253  time  1.9253463745117188
Average time per epoch 1.108683334350586 (for 500 epochs)
Epoch  254  time  1.940312147140503
Average time per epoch 1.112563958644867 (for 500 epochs)
Epoch  255  time  2.2849066257476807
Average time per epoch 1.1171337718963623 (for 500 epochs)
Epoch  256  time  3.27163028717041
Average time per epoch 1.123677032470703 (for 500 epochs)
Epoch  257  time  3.401638984680176
Average time per epoch 1.1304803104400636 (for 500 epochs)
Epoch  258  time  3.3350698947906494
Average time per epoch 1.1371504502296448 (for 500 epochs)
Epoch  259  time  2.30902099609375
Average time per epoch 1.1417684922218323 (for 500 epochs)
Epoch  260  time  1.9703493118286133
Epoch  260  loss  0.2663703769635146 correct 50
Average time per epoch 1.1457091908454895 (for 500 epochs)
Epoch  261  time  1.9316034317016602
Average time per epoch 1.1495723977088927 (for 500 epochs)
Epoch  262  time  2.0159096717834473
Average time per epoch 1.1536042170524596 (for 500 epochs)
Epoch  263  time  1.936814546585083
Average time per epoch 1.1574778461456299 (for 500 epochs)
Epoch  264  time  1.9611961841583252
Average time per epoch 1.1614002385139466 (for 500 epochs)
Epoch  265  time  1.9367449283599854
Average time per epoch 1.1652737283706665 (for 500 epochs)
Epoch  266  time  2.010350465774536
Average time per epoch 1.1692944293022156 (for 500 epochs)
Epoch  267  time  1.926750898361206
Average time per epoch 1.173147931098938 (for 500 epochs)
Epoch  268  time  1.9494895935058594
Average time per epoch 1.1770469102859498 (for 500 epochs)
Epoch  269  time  2.014568328857422
Average time per epoch 1.1810760469436645 (for 500 epochs)
Epoch  270  time  1.9395434856414795
Epoch  270  loss  0.6101444245152504 correct 50
Average time per epoch 1.1849551339149476 (for 500 epochs)
Epoch  271  time  1.9500203132629395
Average time per epoch 1.1888551745414735 (for 500 epochs)
Epoch  272  time  2.016291856765747
Average time per epoch 1.1928877582550048 (for 500 epochs)
Epoch  273  time  1.9649310111999512
Average time per epoch 1.1968176202774048 (for 500 epochs)
Epoch  274  time  1.9069020748138428
Average time per epoch 1.2006314244270324 (for 500 epochs)
Epoch  275  time  1.936654806137085
Average time per epoch 1.2045047340393067 (for 500 epochs)
Epoch  276  time  2.0115907192230225
Average time per epoch 1.2085279154777526 (for 500 epochs)
Epoch  277  time  1.9345860481262207
Average time per epoch 1.212397087574005 (for 500 epochs)
Epoch  278  time  1.9405274391174316
Average time per epoch 1.21627814245224 (for 500 epochs)
Epoch  279  time  2.026451826095581
Average time per epoch 1.2203310461044312 (for 500 epochs)
Epoch  280  time  1.9864327907562256
Epoch  280  loss  0.10926051801891445 correct 50
Average time per epoch 1.2243039116859435 (for 500 epochs)
Epoch  281  time  1.9545280933380127
Average time per epoch 1.2282129678726197 (for 500 epochs)
Epoch  282  time  2.0542044639587402
Average time per epoch 1.2323213768005372 (for 500 epochs)
Epoch  283  time  1.9580998420715332
Average time per epoch 1.2362375764846802 (for 500 epochs)
Epoch  284  time  1.986424446105957
Average time per epoch 1.2402104253768922 (for 500 epochs)
Epoch  285  time  1.9594330787658691
Average time per epoch 1.2441292915344238 (for 500 epochs)
Epoch  286  time  2.044455051422119
Average time per epoch 1.248218201637268 (for 500 epochs)
Epoch  287  time  1.9501278400421143
Average time per epoch 1.2521184573173523 (for 500 epochs)
Epoch  288  time  2.0643186569213867
Average time per epoch 1.256247094631195 (for 500 epochs)
Epoch  289  time  3.2814524173736572
Average time per epoch 1.2628099994659423 (for 500 epochs)
Epoch  290  time  3.374894142150879
Epoch  290  loss  0.10804032915905604 correct 50
Average time per epoch 1.2695597877502443 (for 500 epochs)
Epoch  291  time  3.42596173286438
Average time per epoch 1.276411711215973 (for 500 epochs)
Epoch  292  time  2.445910930633545
Average time per epoch 1.28130353307724 (for 500 epochs)
Epoch  293  time  2.041403293609619
Average time per epoch 1.2853863396644591 (for 500 epochs)
Epoch  294  time  1.9385595321655273
Average time per epoch 1.2892634587287903 (for 500 epochs)
Epoch  295  time  1.9536828994750977
Average time per epoch 1.2931708245277405 (for 500 epochs)
Epoch  296  time  2.007387399673462
Average time per epoch 1.2971855993270873 (for 500 epochs)
Epoch  297  time  1.950782299041748
Average time per epoch 1.3010871639251709 (for 500 epochs)
Epoch  298  time  1.956063985824585
Average time per epoch 1.30499929189682 (for 500 epochs)
Epoch  299  time  2.0542213916778564
Average time per epoch 1.3091077346801758 (for 500 epochs)
Epoch  300  time  1.9413981437683105
Epoch  300  loss  0.2695871530327252 correct 50
Average time per epoch 1.3129905309677123 (for 500 epochs)
Epoch  301  time  2.1803195476531982
Average time per epoch 1.3173511700630187 (for 500 epochs)
Epoch  302  time  3.2166903018951416
Average time per epoch 1.3237845506668091 (for 500 epochs)
Epoch  303  time  3.462317943572998
Average time per epoch 1.330709186553955 (for 500 epochs)
Epoch  304  time  3.4163715839385986
Average time per epoch 1.3375419297218323 (for 500 epochs)
Epoch  305  time  2.4002671241760254
Average time per epoch 1.3423424639701844 (for 500 epochs)
Epoch  306  time  2.007922649383545
Average time per epoch 1.3463583092689515 (for 500 epochs)
Epoch  307  time  1.9446845054626465
Average time per epoch 1.3502476782798767 (for 500 epochs)
Epoch  308  time  1.953688383102417
Average time per epoch 1.3541550550460815 (for 500 epochs)
Epoch  309  time  1.945127248764038
Average time per epoch 1.3580453095436096 (for 500 epochs)
Epoch  310  time  2.0240635871887207
Epoch  310  loss  0.4915057960754495 correct 50
Average time per epoch 1.362093436717987 (for 500 epochs)
Epoch  311  time  1.9417424201965332
Average time per epoch 1.3659769215583801 (for 500 epochs)
Epoch  312  time  1.978039264678955
Average time per epoch 1.369933000087738 (for 500 epochs)
Epoch  313  time  2.0021586418151855
Average time per epoch 1.3739373173713685 (for 500 epochs)
Epoch  314  time  1.9132075309753418
Average time per epoch 1.3777637324333192 (for 500 epochs)
Epoch  315  time  1.9584956169128418
Average time per epoch 1.3816807236671447 (for 500 epochs)
Epoch  316  time  2.044116973876953
Average time per epoch 1.3857689576148986 (for 500 epochs)
Epoch  317  time  1.9617640972137451
Average time per epoch 1.3896924858093263 (for 500 epochs)
Epoch  318  time  1.9537627696990967
Average time per epoch 1.3936000113487244 (for 500 epochs)
Epoch  319  time  1.9541277885437012
Average time per epoch 1.3975082669258119 (for 500 epochs)
Epoch  320  time  2.143615484237671
Epoch  320  loss  0.10647986561153112 correct 50
Average time per epoch 1.401795497894287 (for 500 epochs)
Epoch  321  time  3.20771861076355
Average time per epoch 1.4082109351158143 (for 500 epochs)
Epoch  322  time  3.359194755554199
Average time per epoch 1.4149293246269226 (for 500 epochs)
Epoch  323  time  3.4409191608428955
Average time per epoch 1.4218111629486083 (for 500 epochs)
Epoch  324  time  2.4553563594818115
Average time per epoch 1.426721875667572 (for 500 epochs)
Epoch  325  time  1.9767413139343262
Average time per epoch 1.4306753582954406 (for 500 epochs)
Epoch  326  time  2.0015532970428467
Average time per epoch 1.4346784648895263 (for 500 epochs)
Epoch  327  time  2.00968337059021
Average time per epoch 1.4386978316307069 (for 500 epochs)
Epoch  328  time  2.5215179920196533
Average time per epoch 1.443740867614746 (for 500 epochs)
Epoch  329  time  3.3877906799316406
Average time per epoch 1.4505164489746094 (for 500 epochs)
Epoch  330  time  3.4596965312957764
Epoch  330  loss  0.10777100693614845 correct 50
Average time per epoch 1.457435842037201 (for 500 epochs)
Epoch  331  time  3.2484445571899414
Average time per epoch 1.4639327311515808 (for 500 epochs)
Epoch  332  time  1.9420802593231201
Average time per epoch 1.467816891670227 (for 500 epochs)
Epoch  333  time  1.9992728233337402
Average time per epoch 1.4718154373168946 (for 500 epochs)
Epoch  334  time  1.9665157794952393
Average time per epoch 1.475748468875885 (for 500 epochs)
Epoch  335  time  1.9520113468170166
Average time per epoch 1.479652491569519 (for 500 epochs)
Epoch  336  time  1.954261064529419
Average time per epoch 1.483561013698578 (for 500 epochs)
Epoch  337  time  2.0194091796875
Average time per epoch 1.4875998320579529 (for 500 epochs)
Epoch  338  time  1.968442678451538
Average time per epoch 1.491536717414856 (for 500 epochs)
Epoch  339  time  1.929567813873291
Average time per epoch 1.4953958530426026 (for 500 epochs)
Epoch  340  time  2.0054357051849365
Epoch  340  loss  0.11577010069220409 correct 50
Average time per epoch 1.4994067244529725 (for 500 epochs)
Epoch  341  time  1.9471478462219238
Average time per epoch 1.5033010201454162 (for 500 epochs)
Epoch  342  time  1.9637229442596436
Average time per epoch 1.5072284660339355 (for 500 epochs)
Epoch  343  time  1.9993541240692139
Average time per epoch 1.511227174282074 (for 500 epochs)
Epoch  344  time  1.9558732509613037
Average time per epoch 1.5151389207839965 (for 500 epochs)
Epoch  345  time  1.968127727508545
Average time per epoch 1.5190751762390138 (for 500 epochs)
Epoch  346  time  1.9794678688049316
Average time per epoch 1.5230341119766235 (for 500 epochs)
Epoch  347  time  2.062211513519287
Average time per epoch 1.527158535003662 (for 500 epochs)
Epoch  348  time  1.9366645812988281
Average time per epoch 1.5310318641662597 (for 500 epochs)
Epoch  349  time  1.9420504570007324
Average time per epoch 1.5349159650802613 (for 500 epochs)
Epoch  350  time  2.0325310230255127
Epoch  350  loss  0.07257190728571813 correct 50
Average time per epoch 1.5389810271263122 (for 500 epochs)
Epoch  351  time  1.9580793380737305
Average time per epoch 1.5428971858024598 (for 500 epochs)
Epoch  352  time  2.3010799884796143
Average time per epoch 1.547499345779419 (for 500 epochs)
Epoch  353  time  3.283320188522339
Average time per epoch 1.5540659861564636 (for 500 epochs)
Epoch  354  time  3.5221762657165527
Average time per epoch 1.5611103386878968 (for 500 epochs)
Epoch  355  time  3.3144237995147705
Average time per epoch 1.5677391862869263 (for 500 epochs)
Epoch  356  time  2.2663586139678955
Average time per epoch 1.572271903514862 (for 500 epochs)
Epoch  357  time  1.9990859031677246
Average time per epoch 1.5762700753211976 (for 500 epochs)
Epoch  358  time  1.9722814559936523
Average time per epoch 1.5802146382331848 (for 500 epochs)
Epoch  359  time  1.9240283966064453
Average time per epoch 1.5840626950263976 (for 500 epochs)
Epoch  360  time  2.037130355834961
Epoch  360  loss  0.025891151415888602 correct 50
Average time per epoch 1.5881369557380676 (for 500 epochs)
Epoch  361  time  1.9784324169158936
Average time per epoch 1.5920938205718995 (for 500 epochs)
Epoch  362  time  1.930168628692627
Average time per epoch 1.5959541578292846 (for 500 epochs)
Epoch  363  time  1.915910243988037
Average time per epoch 1.5997859783172608 (for 500 epochs)
Epoch  364  time  2.013209581375122
Average time per epoch 1.603812397480011 (for 500 epochs)
Epoch  365  time  1.9800786972045898
Average time per epoch 1.6077725548744202 (for 500 epochs)
Epoch  366  time  1.9729790687561035
Average time per epoch 1.6117185130119325 (for 500 epochs)
Epoch  367  time  1.9997634887695312
Average time per epoch 1.6157180399894715 (for 500 epochs)
Epoch  368  time  1.9412822723388672
Average time per epoch 1.6196006045341491 (for 500 epochs)
Epoch  369  time  1.9574429988861084
Average time per epoch 1.6235154905319213 (for 500 epochs)
Epoch  370  time  1.9270906448364258
Epoch  370  loss  0.2574792299758383 correct 50
Average time per epoch 1.6273696718215942 (for 500 epochs)
Epoch  371  time  1.9876608848571777
Average time per epoch 1.6313449935913087 (for 500 epochs)
Epoch  372  time  1.9556324481964111
Average time per epoch 1.6352562584877015 (for 500 epochs)
Epoch  373  time  1.9534146785736084
Average time per epoch 1.6391630878448487 (for 500 epochs)
Epoch  374  time  2.015676975250244
Average time per epoch 1.643194441795349 (for 500 epochs)
Epoch  375  time  1.9805080890655518
Average time per epoch 1.6471554579734802 (for 500 epochs)
Epoch  376  time  1.9357447624206543
Average time per epoch 1.6510269474983215 (for 500 epochs)
Epoch  377  time  2.0103678703308105
Average time per epoch 1.655047683238983 (for 500 epochs)
Epoch  378  time  1.9192628860473633
Average time per epoch 1.6588862090110779 (for 500 epochs)
Epoch  379  time  1.922943353652954
Average time per epoch 1.6627320957183838 (for 500 epochs)
Epoch  380  time  1.9280681610107422
Epoch  380  loss  0.12271950583360838 correct 50
Average time per epoch 1.6665882320404053 (for 500 epochs)
Epoch  381  time  2.023494005203247
Average time per epoch 1.6706352200508117 (for 500 epochs)
Epoch  382  time  1.9938538074493408
Average time per epoch 1.6746229276657105 (for 500 epochs)
Epoch  383  time  1.995004415512085
Average time per epoch 1.6786129364967346 (for 500 epochs)
Epoch  384  time  2.014211654663086
Average time per epoch 1.6826413598060608 (for 500 epochs)
Epoch  385  time  1.9775004386901855
Average time per epoch 1.686596360683441 (for 500 epochs)
Epoch  386  time  3.13826584815979
Average time per epoch 1.6928728923797607 (for 500 epochs)
Epoch  387  time  3.369359016418457
Average time per epoch 1.6996116104125976 (for 500 epochs)
Epoch  388  time  3.5078535079956055
Average time per epoch 1.7066273174285889 (for 500 epochs)
Epoch  389  time  2.651916980743408
Average time per epoch 1.7119311513900757 (for 500 epochs)
Epoch  390  time  1.9625329971313477
Epoch  390  loss  0.1308878202766096 correct 50
Average time per epoch 1.7158562173843384 (for 500 epochs)
Epoch  391  time  2.041049003601074
Average time per epoch 1.7199383153915406 (for 500 epochs)
Epoch  392  time  1.9659233093261719
Average time per epoch 1.7238701620101928 (for 500 epochs)
Epoch  393  time  1.9691057205200195
Average time per epoch 1.7278083734512328 (for 500 epochs)
Epoch  394  time  2.053138256072998
Average time per epoch 1.7319146499633788 (for 500 epochs)
Epoch  395  time  1.9556007385253906
Average time per epoch 1.7358258514404297 (for 500 epochs)
Epoch  396  time  1.9477479457855225
Average time per epoch 1.7397213473320008 (for 500 epochs)
Epoch  397  time  1.954329252243042
Average time per epoch 1.7436300058364869 (for 500 epochs)
Epoch  398  time  2.010274887084961
Average time per epoch 1.7476505556106567 (for 500 epochs)
Epoch  399  time  1.960975170135498
Average time per epoch 1.7515725059509277 (for 500 epochs)
Epoch  400  time  1.9711015224456787
Epoch  400  loss  0.13622695338280272 correct 50
Average time per epoch 1.7555147089958192 (for 500 epochs)
Epoch  401  time  2.007128953933716
Average time per epoch 1.7595289669036864 (for 500 epochs)
Epoch  402  time  1.9401865005493164
Average time per epoch 1.7634093399047852 (for 500 epochs)
Epoch  403  time  1.9622645378112793
Average time per epoch 1.7673338689804077 (for 500 epochs)
Epoch  404  time  2.0162155628204346
Average time per epoch 1.7713663001060487 (for 500 epochs)
Epoch  405  time  1.9207160472869873
Average time per epoch 1.7752077322006226 (for 500 epochs)
Epoch  406  time  1.9313380718231201
Average time per epoch 1.7790704083442688 (for 500 epochs)
Epoch  407  time  1.9498729705810547
Average time per epoch 1.7829701542854308 (for 500 epochs)
Epoch  408  time  1.9977374076843262
Average time per epoch 1.7869656291007996 (for 500 epochs)
Epoch  409  time  1.9241011142730713
Average time per epoch 1.7908138313293458 (for 500 epochs)
Epoch  410  time  1.9639298915863037
Epoch  410  loss  0.05261342228884254 correct 50
Average time per epoch 1.7947416911125182 (for 500 epochs)
Epoch  411  time  2.037687063217163
Average time per epoch 1.7988170652389526 (for 500 epochs)
Epoch  412  time  1.9627034664154053
Average time per epoch 1.8027424721717835 (for 500 epochs)
Epoch  413  time  1.940582036972046
Average time per epoch 1.8066236362457275 (for 500 epochs)
Epoch  414  time  1.9169049263000488
Average time per epoch 1.8104574460983276 (for 500 epochs)
Epoch  415  time  1.9932737350463867
Average time per epoch 1.8144439935684205 (for 500 epochs)
Epoch  416  time  1.9556334018707275
Average time per epoch 1.8183552603721618 (for 500 epochs)
Epoch  417  time  1.9445302486419678
Average time per epoch 1.8222443208694459 (for 500 epochs)
Epoch  418  time  2.0393433570861816
Average time per epoch 1.826323007583618 (for 500 epochs)
Epoch  419  time  1.946242094039917
Average time per epoch 1.830215491771698 (for 500 epochs)
Epoch  420  time  3.1511292457580566
Epoch  420  loss  0.1487497221756619 correct 50
Average time per epoch 1.8365177502632142 (for 500 epochs)
Epoch  421  time  3.4301984310150146
Average time per epoch 1.8433781471252442 (for 500 epochs)
Epoch  422  time  3.379674196243286
Average time per epoch 1.8501374955177308 (for 500 epochs)
Epoch  423  time  2.585638999938965
Average time per epoch 1.8553087735176086 (for 500 epochs)
Epoch  424  time  1.9487841129302979
Average time per epoch 1.8592063417434692 (for 500 epochs)
Epoch  425  time  2.0236990451812744
Average time per epoch 1.8632537398338318 (for 500 epochs)
Epoch  426  time  1.9717614650726318
Average time per epoch 1.867197262763977 (for 500 epochs)
Epoch  427  time  1.9457461833953857
Average time per epoch 1.8710887551307678 (for 500 epochs)
Epoch  428  time  2.039313793182373
Average time per epoch 1.8751673827171325 (for 500 epochs)
Epoch  429  time  1.962813138961792
Average time per epoch 1.8790930089950562 (for 500 epochs)
Epoch  430  time  1.9609744548797607
Epoch  430  loss  0.07267798666724941 correct 50
Average time per epoch 1.8830149579048157 (for 500 epochs)
Epoch  431  time  1.9485232830047607
Average time per epoch 1.8869120044708252 (for 500 epochs)
Epoch  432  time  1.9920053482055664
Average time per epoch 1.8908960151672363 (for 500 epochs)
Epoch  433  time  1.972179651260376
Average time per epoch 1.894840374469757 (for 500 epochs)
Epoch  434  time  1.9835624694824219
Average time per epoch 1.8988074994087218 (for 500 epochs)
Epoch  435  time  2.005242347717285
Average time per epoch 1.9028179841041566 (for 500 epochs)
Epoch  436  time  1.958841323852539
Average time per epoch 1.9067356667518616 (for 500 epochs)
Epoch  437  time  1.9501662254333496
Average time per epoch 1.9106359992027282 (for 500 epochs)
Epoch  438  time  2.293015241622925
Average time per epoch 1.915222029685974 (for 500 epochs)
Epoch  439  time  3.277690887451172
Average time per epoch 1.9217774114608766 (for 500 epochs)
Epoch  440  time  3.3451359272003174
Epoch  440  loss  0.017605340632783423 correct 50
Average time per epoch 1.9284676833152772 (for 500 epochs)
Epoch  441  time  3.3401355743408203
Average time per epoch 1.9351479544639587 (for 500 epochs)
Epoch  442  time  2.3163111209869385
Average time per epoch 1.9397805767059326 (for 500 epochs)
Epoch  443  time  1.9539854526519775
Average time per epoch 1.9436885476112367 (for 500 epochs)
Epoch  444  time  1.9402949810028076
Average time per epoch 1.9475691375732422 (for 500 epochs)
Epoch  445  time  2.0124385356903076
Average time per epoch 1.951594014644623 (for 500 epochs)
Epoch  446  time  1.9620962142944336
Average time per epoch 1.9555182070732118 (for 500 epochs)
Epoch  447  time  1.9421191215515137
Average time per epoch 1.9594024453163148 (for 500 epochs)
Epoch  448  time  1.9971208572387695
Average time per epoch 1.9633966870307922 (for 500 epochs)
Epoch  449  time  1.9719202518463135
Average time per epoch 1.9673405275344849 (for 500 epochs)
Epoch  450  time  2.076709270477295
Epoch  450  loss  0.1366863591795564 correct 50
Average time per epoch 1.9714939460754395 (for 500 epochs)
Epoch  451  time  3.198779344558716
Average time per epoch 1.9778915047645569 (for 500 epochs)
Epoch  452  time  3.4681687355041504
Average time per epoch 1.9848278422355652 (for 500 epochs)
Epoch  453  time  3.343148946762085
Average time per epoch 1.9915141401290894 (for 500 epochs)
Epoch  454  time  2.4346883296966553
Average time per epoch 1.9963835167884827 (for 500 epochs)
Epoch  455  time  1.9961583614349365
Average time per epoch 2.0003758335113524 (for 500 epochs)
Epoch  456  time  1.9957456588745117
Average time per epoch 2.0043673248291016 (for 500 epochs)
Epoch  457  time  1.9821860790252686
Average time per epoch 2.008331696987152 (for 500 epochs)
Epoch  458  time  1.9280281066894531
Average time per epoch 2.0121877532005312 (for 500 epochs)
Epoch  459  time  2.010948657989502
Average time per epoch 2.01620965051651 (for 500 epochs)
Epoch  460  time  1.9036900997161865
Epoch  460  loss  0.06577980575570361 correct 50
Average time per epoch 2.0200170307159424 (for 500 epochs)
Epoch  461  time  1.9172372817993164
Average time per epoch 2.023851505279541 (for 500 epochs)
Epoch  462  time  2.012753486633301
Average time per epoch 2.027877012252808 (for 500 epochs)
Epoch  463  time  1.9344837665557861
Average time per epoch 2.031745979785919 (for 500 epochs)
Epoch  464  time  2.018439292907715
Average time per epoch 2.035782858371735 (for 500 epochs)
Epoch  465  time  2.9221088886260986
Average time per epoch 2.0416270761489868 (for 500 epochs)
Epoch  466  time  3.4377152919769287
Average time per epoch 2.0485025067329405 (for 500 epochs)
Epoch  467  time  3.3738455772399902
Average time per epoch 2.0552501978874207 (for 500 epochs)
Epoch  468  time  2.8668062686920166
Average time per epoch 2.060983810424805 (for 500 epochs)
Epoch  469  time  2.00115966796875
Average time per epoch 2.0649861297607424 (for 500 epochs)
Epoch  470  time  1.9425795078277588
Epoch  470  loss  0.08995872168829651 correct 50
Average time per epoch 2.0688712887763976 (for 500 epochs)
Epoch  471  time  1.9074454307556152
Average time per epoch 2.072686179637909 (for 500 epochs)
Epoch  472  time  2.0103893280029297
Average time per epoch 2.0767069582939146 (for 500 epochs)
Epoch  473  time  1.9521739482879639
Average time per epoch 2.0806113061904905 (for 500 epochs)
Epoch  474  time  1.9303297996520996
Average time per epoch 2.084471965789795 (for 500 epochs)
Epoch  475  time  1.9459447860717773
Average time per epoch 2.0883638553619384 (for 500 epochs)
Epoch  476  time  1.9877574443817139
Average time per epoch 2.092339370250702 (for 500 epochs)
Epoch  477  time  1.9700636863708496
Average time per epoch 2.096279497623444 (for 500 epochs)
Epoch  478  time  1.949594259262085
Average time per epoch 2.100178686141968 (for 500 epochs)
Epoch  479  time  2.0041816234588623
Average time per epoch 2.1041870493888855 (for 500 epochs)
Epoch  480  time  1.9378461837768555
Epoch  480  loss  0.021113250685904673 correct 50
Average time per epoch 2.1080627417564393 (for 500 epochs)
Epoch  481  time  1.9354093074798584
Average time per epoch 2.111933560371399 (for 500 epochs)
Epoch  482  time  2.1287527084350586
Average time per epoch 2.116191065788269 (for 500 epochs)
Epoch  483  time  3.1817169189453125
Average time per epoch 2.1225544996261596 (for 500 epochs)
Epoch  484  time  3.301978349685669
Average time per epoch 2.129158456325531 (for 500 epochs)
Epoch  485  time  3.3549301624298096
Average time per epoch 2.1358683166503907 (for 500 epochs)
Epoch  486  time  2.63461971282959
Average time per epoch 2.1411375560760497 (for 500 epochs)
Epoch  487  time  1.9587440490722656
Average time per epoch 2.1450550441741942 (for 500 epochs)
Epoch  488  time  1.9416706562042236
Average time per epoch 2.148938385486603 (for 500 epochs)
Epoch  489  time  2.00666880607605
Average time per epoch 2.152951723098755 (for 500 epochs)
Epoch  490  time  2.0057311058044434
Epoch  490  loss  0.1264751998384887 correct 50
Average time per epoch 2.1569631853103637 (for 500 epochs)
Epoch  491  time  1.935723066329956
Average time per epoch 2.1608346314430236 (for 500 epochs)
Epoch  492  time  1.9857752323150635
Average time per epoch 2.164806181907654 (for 500 epochs)
Epoch  493  time  2.0360043048858643
Average time per epoch 2.1688781905174257 (for 500 epochs)
Epoch  494  time  1.9656691551208496
Average time per epoch 2.1728095288276674 (for 500 epochs)
Epoch  495  time  1.9919695854187012
Average time per epoch 2.176793467998505 (for 500 epochs)
Epoch  496  time  2.016552209854126
Average time per epoch 2.180826572418213 (for 500 epochs)
Epoch  497  time  1.9455101490020752
Average time per epoch 2.184717592716217 (for 500 epochs)
Epoch  498  time  1.9708807468414307
Average time per epoch 2.1886593542099 (for 500 epochs)
Epoch  499  time  2.013315439224243
Average time per epoch 2.192685985088348 (for 500 epochs)

```
# GPU XOR Dataset:
```
Epoch  0  loss  6.538789462487502 correct 30
Average time per epoch 0.008162228584289551 (for 500 epochs)
Epoch  1  time  2.0950076580047607
Average time per epoch 0.012352243900299072 (for 500 epochs)
Epoch  2  time  2.02783203125
Average time per epoch 0.016407907962799072 (for 500 epochs)
Epoch  3  time  2.055367946624756
Average time per epoch 0.020518643856048582 (for 500 epochs)
Epoch  4  time  2.0344622135162354
Average time per epoch 0.024587568283081056 (for 500 epochs)
Epoch  5  time  2.080280065536499
Average time per epoch 0.028748128414154053 (for 500 epochs)
Epoch  6  time  2.001180410385132
Average time per epoch 0.03275048923492432 (for 500 epochs)
Epoch  7  time  2.024249792098999
Average time per epoch 0.036798988819122316 (for 500 epochs)
Epoch  8  time  2.1612820625305176
Average time per epoch 0.04112155294418335 (for 500 epochs)
Epoch  9  time  2.0381765365600586
Average time per epoch 0.04519790601730347 (for 500 epochs)
Epoch  10  time  2.0348660945892334
Epoch  10  loss  5.782273204182897 correct 26
Average time per epoch 0.04926763820648193 (for 500 epochs)
Epoch  11  time  2.4021780490875244
Average time per epoch 0.054071994304656985 (for 500 epochs)
Epoch  12  time  3.466336727142334
Average time per epoch 0.06100466775894165 (for 500 epochs)
Epoch  13  time  3.460622787475586
Average time per epoch 0.06792591333389282 (for 500 epochs)
Epoch  14  time  3.432530641555786
Average time per epoch 0.07479097461700439 (for 500 epochs)
Epoch  15  time  2.1863579750061035
Average time per epoch 0.0791636905670166 (for 500 epochs)
Epoch  16  time  2.018666982650757
Average time per epoch 0.08320102453231812 (for 500 epochs)
Epoch  17  time  2.031968116760254
Average time per epoch 0.08726496076583862 (for 500 epochs)
Epoch  18  time  2.072693347930908
Average time per epoch 0.09141034746170044 (for 500 epochs)
Epoch  19  time  2.0193378925323486
Average time per epoch 0.09544902324676513 (for 500 epochs)
Epoch  20  time  2.0773372650146484
Epoch  20  loss  5.22208027295506 correct 43
Average time per epoch 0.09960369777679444 (for 500 epochs)
Epoch  21  time  2.0761489868164062
Average time per epoch 0.10375599575042725 (for 500 epochs)
Epoch  22  time  2.078322410583496
Average time per epoch 0.10791264057159423 (for 500 epochs)
Epoch  23  time  2.0194497108459473
Average time per epoch 0.11195153999328614 (for 500 epochs)
Epoch  24  time  2.006854772567749
Average time per epoch 0.11596524953842163 (for 500 epochs)
Epoch  25  time  2.082371950149536
Average time per epoch 0.1201299934387207 (for 500 epochs)
Epoch  26  time  2.1274120807647705
Average time per epoch 0.12438481760025025 (for 500 epochs)
Epoch  27  time  3.3326287269592285
Average time per epoch 0.1310500750541687 (for 500 epochs)
Epoch  28  time  3.5209789276123047
Average time per epoch 0.13809203290939331 (for 500 epochs)
Epoch  29  time  3.6418874263763428
Average time per epoch 0.145375807762146 (for 500 epochs)
Epoch  30  time  2.4086432456970215
Epoch  30  loss  7.598469087424302 correct 31
Average time per epoch 0.15019309425354005 (for 500 epochs)
Epoch  31  time  2.356147289276123
Average time per epoch 0.1549053888320923 (for 500 epochs)
Epoch  32  time  3.56296443939209
Average time per epoch 0.16203131771087648 (for 500 epochs)
Epoch  33  time  3.5469367504119873
Average time per epoch 0.16912519121170044 (for 500 epochs)
Epoch  34  time  3.523434638977051
Average time per epoch 0.17617206048965453 (for 500 epochs)
Epoch  35  time  2.1452667713165283
Average time per epoch 0.1804625940322876 (for 500 epochs)
Epoch  36  time  2.046156644821167
Average time per epoch 0.18455490732192995 (for 500 epochs)
Epoch  37  time  2.0195581912994385
Average time per epoch 0.1885940237045288 (for 500 epochs)
Epoch  38  time  2.0274360179901123
Average time per epoch 0.19264889574050903 (for 500 epochs)
Epoch  39  time  2.0750720500946045
Average time per epoch 0.19679903984069824 (for 500 epochs)
Epoch  40  time  1.9647488594055176
Epoch  40  loss  5.524672601750655 correct 42
Average time per epoch 0.20072853755950928 (for 500 epochs)
Epoch  41  time  2.0073325634002686
Average time per epoch 0.20474320268630983 (for 500 epochs)
Epoch  42  time  2.1067559719085693
Average time per epoch 0.20895671463012697 (for 500 epochs)
Epoch  43  time  2.0085484981536865
Average time per epoch 0.21297381162643433 (for 500 epochs)
Epoch  44  time  1.9913334846496582
Average time per epoch 0.21695647859573364 (for 500 epochs)
Epoch  45  time  2.1572093963623047
Average time per epoch 0.22127089738845826 (for 500 epochs)
Epoch  46  time  2.0483880043029785
Average time per epoch 0.2253676733970642 (for 500 epochs)
Epoch  47  time  2.0245399475097656
Average time per epoch 0.22941675329208375 (for 500 epochs)
Epoch  48  time  2.0020980834960938
Average time per epoch 0.23342094945907593 (for 500 epochs)
Epoch  49  time  2.0675768852233887
Average time per epoch 0.2375561032295227 (for 500 epochs)
Epoch  50  time  2.0334298610687256
Epoch  50  loss  2.43319724836471 correct 38
Average time per epoch 0.24162296295166016 (for 500 epochs)
Epoch  51  time  2.0090601444244385
Average time per epoch 0.24564108324050904 (for 500 epochs)
Epoch  52  time  2.0935745239257812
Average time per epoch 0.2498282322883606 (for 500 epochs)
Epoch  53  time  1.9846093654632568
Average time per epoch 0.25379745101928713 (for 500 epochs)
Epoch  54  time  2.0143468379974365
Average time per epoch 0.257826144695282 (for 500 epochs)
Epoch  55  time  2.0022470951080322
Average time per epoch 0.26183063888549807 (for 500 epochs)
Epoch  56  time  2.066227912902832
Average time per epoch 0.2659630947113037 (for 500 epochs)
Epoch  57  time  2.4444541931152344
Average time per epoch 0.27085200309753416 (for 500 epochs)
Epoch  58  time  3.4296648502349854
Average time per epoch 0.27771133279800414 (for 500 epochs)
Epoch  59  time  3.6320419311523438
Average time per epoch 0.28497541666030884 (for 500 epochs)
Epoch  60  time  3.413184642791748
Epoch  60  loss  4.482093558977024 correct 46
Average time per epoch 0.29180178594589234 (for 500 epochs)
Epoch  61  time  2.0425174236297607
Average time per epoch 0.2958868207931519 (for 500 epochs)
Epoch  62  time  2.1402883529663086
Average time per epoch 0.3001673974990845 (for 500 epochs)
Epoch  63  time  2.048217296600342
Average time per epoch 0.30426383209228514 (for 500 epochs)
Epoch  64  time  2.0848774909973145
Average time per epoch 0.30843358707427976 (for 500 epochs)
Epoch  65  time  2.013272523880005
Average time per epoch 0.31246013212203977 (for 500 epochs)
Epoch  66  time  2.051820755004883
Average time per epoch 0.31656377363204957 (for 500 epochs)
Epoch  67  time  2.0096333026885986
Average time per epoch 0.32058304023742673 (for 500 epochs)
Epoch  68  time  1.9888975620269775
Average time per epoch 0.3245608353614807 (for 500 epochs)
Epoch  69  time  2.0795273780822754
Average time per epoch 0.32871989011764524 (for 500 epochs)
Epoch  70  time  2.0212180614471436
Epoch  70  loss  2.0318304931212143 correct 44
Average time per epoch 0.33276232624053953 (for 500 epochs)
Epoch  71  time  2.02055287361145
Average time per epoch 0.33680343198776247 (for 500 epochs)
Epoch  72  time  2.091937303543091
Average time per epoch 0.34098730659484866 (for 500 epochs)
Epoch  73  time  2.0810890197753906
Average time per epoch 0.34514948463439943 (for 500 epochs)
Epoch  74  time  2.0621047019958496
Average time per epoch 0.3492736940383911 (for 500 epochs)
Epoch  75  time  2.0172669887542725
Average time per epoch 0.3533082280158997 (for 500 epochs)
Epoch  76  time  2.0722858905792236
Average time per epoch 0.35745279979705813 (for 500 epochs)
Epoch  77  time  1.9901654720306396
Average time per epoch 0.3614331307411194 (for 500 epochs)
Epoch  78  time  1.9975006580352783
Average time per epoch 0.36542813205718994 (for 500 epochs)
Epoch  79  time  2.0521364212036133
Average time per epoch 0.36953240489959716 (for 500 epochs)
Epoch  80  time  1.986248254776001
Epoch  80  loss  1.694002456321043 correct 47
Average time per epoch 0.37350490140914916 (for 500 epochs)
Epoch  81  time  1.9849333763122559
Average time per epoch 0.37747476816177367 (for 500 epochs)
Epoch  82  time  2.0095088481903076
Average time per epoch 0.3814937858581543 (for 500 epochs)
Epoch  83  time  2.0410773754119873
Average time per epoch 0.38557594060897826 (for 500 epochs)
Epoch  84  time  1.9780495166778564
Average time per epoch 0.38953203964233396 (for 500 epochs)
Epoch  85  time  1.9745838642120361
Average time per epoch 0.39348120737075803 (for 500 epochs)
Epoch  86  time  2.0707497596740723
Average time per epoch 0.3976227068901062 (for 500 epochs)
Epoch  87  time  2.005234718322754
Average time per epoch 0.4016331763267517 (for 500 epochs)
Epoch  88  time  1.9638373851776123
Average time per epoch 0.40556085109710693 (for 500 epochs)
Epoch  89  time  2.090334892272949
Average time per epoch 0.40974152088165283 (for 500 epochs)
Epoch  90  time  1.992574691772461
Epoch  90  loss  1.3941165298416747 correct 47
Average time per epoch 0.41372667026519777 (for 500 epochs)
Epoch  91  time  2.625535488128662
Average time per epoch 0.4189777412414551 (for 500 epochs)
Epoch  92  time  3.4167349338531494
Average time per epoch 0.42581121110916137 (for 500 epochs)
Epoch  93  time  3.5591959953308105
Average time per epoch 0.432929603099823 (for 500 epochs)
Epoch  94  time  3.2070565223693848
Average time per epoch 0.43934371614456175 (for 500 epochs)
Epoch  95  time  1.984403371810913
Average time per epoch 0.44331252288818357 (for 500 epochs)
Epoch  96  time  2.0544087886810303
Average time per epoch 0.44742134046554566 (for 500 epochs)
Epoch  97  time  2.023097276687622
Average time per epoch 0.4514675350189209 (for 500 epochs)
Epoch  98  time  1.99263334274292
Average time per epoch 0.4554528017044067 (for 500 epochs)
Epoch  99  time  1.9944753646850586
Average time per epoch 0.45944175243377683 (for 500 epochs)
Epoch  100  time  2.0978264808654785
Epoch  100  loss  2.442540827113062 correct 48
Average time per epoch 0.4636374053955078 (for 500 epochs)
Epoch  101  time  1.9792163372039795
Average time per epoch 0.4675958380699158 (for 500 epochs)
Epoch  102  time  2.0556159019470215
Average time per epoch 0.47170706987380984 (for 500 epochs)
Epoch  103  time  2.0611939430236816
Average time per epoch 0.4758294577598572 (for 500 epochs)
Epoch  104  time  2.0376646518707275
Average time per epoch 0.4799047870635986 (for 500 epochs)
Epoch  105  time  1.9905898571014404
Average time per epoch 0.4838859667778015 (for 500 epochs)
Epoch  106  time  2.0729427337646484
Average time per epoch 0.4880318522453308 (for 500 epochs)
Epoch  107  time  2.0042505264282227
Average time per epoch 0.49204035329818724 (for 500 epochs)
Epoch  108  time  2.0071866512298584
Average time per epoch 0.496054726600647 (for 500 epochs)
Epoch  109  time  2.0073978900909424
Average time per epoch 0.5000695223808289 (for 500 epochs)
Epoch  110  time  2.0484020709991455
Epoch  110  loss  2.6590366022676752 correct 48
Average time per epoch 0.5041663265228271 (for 500 epochs)
Epoch  111  time  2.0270063877105713
Average time per epoch 0.5082203392982483 (for 500 epochs)
Epoch  112  time  1.9969885349273682
Average time per epoch 0.512214316368103 (for 500 epochs)
Epoch  113  time  2.029949188232422
Average time per epoch 0.5162742147445679 (for 500 epochs)
Epoch  114  time  1.9685089588165283
Average time per epoch 0.5202112326622009 (for 500 epochs)
Epoch  115  time  1.975581169128418
Average time per epoch 0.5241623950004578 (for 500 epochs)
Epoch  116  time  2.0156867504119873
Average time per epoch 0.5281937685012817 (for 500 epochs)
Epoch  117  time  2.004228115081787
Average time per epoch 0.5322022247314453 (for 500 epochs)
Epoch  118  time  1.9862501621246338
Average time per epoch 0.5361747250556946 (for 500 epochs)
Epoch  119  time  1.9718239307403564
Average time per epoch 0.5401183729171752 (for 500 epochs)
Epoch  120  time  2.0372984409332275
Epoch  120  loss  1.3246308679020855 correct 49
Average time per epoch 0.5441929697990417 (for 500 epochs)
Epoch  121  time  1.9728379249572754
Average time per epoch 0.5481386456489563 (for 500 epochs)
Epoch  122  time  1.9801394939422607
Average time per epoch 0.5520989246368408 (for 500 epochs)
Epoch  123  time  2.034039258956909
Average time per epoch 0.5561670031547546 (for 500 epochs)
Epoch  124  time  1.9603703022003174
Average time per epoch 0.5600877437591553 (for 500 epochs)
Epoch  125  time  2.491199254989624
Average time per epoch 0.5650701422691345 (for 500 epochs)
Epoch  126  time  3.383563995361328
Average time per epoch 0.5718372702598572 (for 500 epochs)
Epoch  127  time  3.5077126026153564
Average time per epoch 0.5788526954650879 (for 500 epochs)
Epoch  128  time  3.3680076599121094
Average time per epoch 0.5855887107849121 (for 500 epochs)
Epoch  129  time  2.021145820617676
Average time per epoch 0.5896310024261474 (for 500 epochs)
Epoch  130  time  2.0873026847839355
Epoch  130  loss  2.2937830303601263 correct 48
Average time per epoch 0.5938056077957153 (for 500 epochs)
Epoch  131  time  2.0104074478149414
Average time per epoch 0.5978264226913452 (for 500 epochs)
Epoch  132  time  2.005244731903076
Average time per epoch 0.6018369121551513 (for 500 epochs)
Epoch  133  time  2.0430796146392822
Average time per epoch 0.6059230713844299 (for 500 epochs)
Epoch  134  time  1.9723305702209473
Average time per epoch 0.6098677325248718 (for 500 epochs)
Epoch  135  time  1.9963898658752441
Average time per epoch 0.6138605122566223 (for 500 epochs)
Epoch  136  time  1.995680332183838
Average time per epoch 0.61785187292099 (for 500 epochs)
Epoch  137  time  2.038419485092163
Average time per epoch 0.6219287118911743 (for 500 epochs)
Epoch  138  time  2.0004727840423584
Average time per epoch 0.625929657459259 (for 500 epochs)
Epoch  139  time  2.0446698665618896
Average time per epoch 0.6300189971923829 (for 500 epochs)
Epoch  140  time  2.0693907737731934
Epoch  140  loss  2.1725424536028872 correct 48
Average time per epoch 0.6341577787399292 (for 500 epochs)
Epoch  141  time  1.971900224685669
Average time per epoch 0.6381015791893005 (for 500 epochs)
Epoch  142  time  1.9802606105804443
Average time per epoch 0.6420621004104614 (for 500 epochs)
Epoch  143  time  2.0125668048858643
Average time per epoch 0.6460872340202332 (for 500 epochs)
Epoch  144  time  2.07788348197937
Average time per epoch 0.6502430009841919 (for 500 epochs)
Epoch  145  time  3.1133296489715576
Average time per epoch 0.656469660282135 (for 500 epochs)
Epoch  146  time  3.4791173934936523
Average time per epoch 0.6634278950691224 (for 500 epochs)
Epoch  147  time  3.6031100749969482
Average time per epoch 0.6706341152191162 (for 500 epochs)
Epoch  148  time  2.700502872467041
Average time per epoch 0.6760351209640503 (for 500 epochs)
Epoch  149  time  1.9604640007019043
Average time per epoch 0.6799560489654541 (for 500 epochs)
Epoch  150  time  2.0351860523223877
Epoch  150  loss  1.5577145569221544 correct 49
Average time per epoch 0.6840264210700989 (for 500 epochs)
Epoch  151  time  2.0015578269958496
Average time per epoch 0.6880295367240906 (for 500 epochs)
Epoch  152  time  1.9704184532165527
Average time per epoch 0.6919703736305237 (for 500 epochs)
Epoch  153  time  1.9664206504821777
Average time per epoch 0.695903214931488 (for 500 epochs)
Epoch  154  time  2.0350329875946045
Average time per epoch 0.6999732809066772 (for 500 epochs)
Epoch  155  time  2.0192339420318604
Average time per epoch 0.704011748790741 (for 500 epochs)
Epoch  156  time  2.0237157344818115
Average time per epoch 0.7080591802597046 (for 500 epochs)
Epoch  157  time  3.3896644115448
Average time per epoch 0.7148385090827942 (for 500 epochs)
Epoch  158  time  3.4123449325561523
Average time per epoch 0.7216631989479065 (for 500 epochs)
Epoch  159  time  3.4567017555236816
Average time per epoch 0.7285766024589538 (for 500 epochs)
Epoch  160  time  2.6141514778137207
Epoch  160  loss  2.8767942018325656 correct 45
Average time per epoch 0.7338049054145813 (for 500 epochs)
Epoch  161  time  2.002657413482666
Average time per epoch 0.7378102202415466 (for 500 epochs)
Epoch  162  time  1.9837262630462646
Average time per epoch 0.7417776727676392 (for 500 epochs)
Epoch  163  time  1.9864590167999268
Average time per epoch 0.7457505908012391 (for 500 epochs)
Epoch  164  time  2.051180362701416
Average time per epoch 0.7498529515266419 (for 500 epochs)
Epoch  165  time  1.9958291053771973
Average time per epoch 0.7538446097373962 (for 500 epochs)
Epoch  166  time  1.9838790893554688
Average time per epoch 0.7578123679161072 (for 500 epochs)
Epoch  167  time  2.961071491241455
Average time per epoch 0.7637345108985901 (for 500 epochs)
Epoch  168  time  3.4272401332855225
Average time per epoch 0.7705889911651611 (for 500 epochs)
Epoch  169  time  3.4746646881103516
Average time per epoch 0.7775383205413818 (for 500 epochs)
Epoch  170  time  2.9048030376434326
Epoch  170  loss  1.6244105126294104 correct 49
Average time per epoch 0.7833479266166687 (for 500 epochs)
Epoch  171  time  2.0343241691589355
Average time per epoch 0.7874165749549866 (for 500 epochs)
Epoch  172  time  2.054827928543091
Average time per epoch 0.7915262308120727 (for 500 epochs)
Epoch  173  time  1.993260145187378
Average time per epoch 0.7955127511024475 (for 500 epochs)
Epoch  174  time  2.044236421585083
Average time per epoch 0.7996012239456177 (for 500 epochs)
Epoch  175  time  1.9901130199432373
Average time per epoch 0.8035814499855042 (for 500 epochs)
Epoch  176  time  2.0123181343078613
Average time per epoch 0.8076060862541199 (for 500 epochs)
Epoch  177  time  2.0558571815490723
Average time per epoch 0.811717800617218 (for 500 epochs)
Epoch  178  time  2.019172191619873
Average time per epoch 0.8157561450004578 (for 500 epochs)
Epoch  179  time  1.9855284690856934
Average time per epoch 0.8197272019386291 (for 500 epochs)
Epoch  180  time  2.0095927715301514
Epoch  180  loss  2.5495362171678084 correct 48
Average time per epoch 0.8237463874816895 (for 500 epochs)
Epoch  181  time  2.0600802898406982
Average time per epoch 0.8278665480613708 (for 500 epochs)
Epoch  182  time  2.021394729614258
Average time per epoch 0.8319093375205994 (for 500 epochs)
Epoch  183  time  2.040301561355591
Average time per epoch 0.8359899406433106 (for 500 epochs)
Epoch  184  time  2.0345053672790527
Average time per epoch 0.8400589513778687 (for 500 epochs)
Epoch  185  time  1.979525089263916
Average time per epoch 0.8440180015563965 (for 500 epochs)
Epoch  186  time  1.9664709568023682
Average time per epoch 0.8479509434700012 (for 500 epochs)
Epoch  187  time  2.0246682167053223
Average time per epoch 0.8520002799034119 (for 500 epochs)
Epoch  188  time  2.2525861263275146
Average time per epoch 0.8565054521560669 (for 500 epochs)
Epoch  189  time  3.284604549407959
Average time per epoch 0.8630746612548829 (for 500 epochs)
Epoch  190  time  3.4284982681274414
Epoch  190  loss  2.0559061481249223 correct 49
Average time per epoch 0.8699316577911377 (for 500 epochs)
Epoch  191  time  3.5066375732421875
Average time per epoch 0.8769449329376221 (for 500 epochs)
Epoch  192  time  2.2010579109191895
Average time per epoch 0.8813470487594605 (for 500 epochs)
Epoch  193  time  1.9626386165618896
Average time per epoch 0.8852723259925842 (for 500 epochs)
Epoch  194  time  2.021343231201172
Average time per epoch 0.8893150124549866 (for 500 epochs)
Epoch  195  time  1.9865868091583252
Average time per epoch 0.8932881860733032 (for 500 epochs)
Epoch  196  time  2.1187450885772705
Average time per epoch 0.8975256762504578 (for 500 epochs)
Epoch  197  time  1.984464168548584
Average time per epoch 0.9014946045875549 (for 500 epochs)
Epoch  198  time  2.0449984073638916
Average time per epoch 0.9055846014022827 (for 500 epochs)
Epoch  199  time  1.9801015853881836
Average time per epoch 0.9095448045730591 (for 500 epochs)
Epoch  200  time  1.9779744148254395
Epoch  200  loss  0.9892408263037399 correct 49
Average time per epoch 0.91350075340271 (for 500 epochs)
Epoch  201  time  2.0472099781036377
Average time per epoch 0.9175951733589173 (for 500 epochs)
Epoch  202  time  1.9585087299346924
Average time per epoch 0.9215121908187867 (for 500 epochs)
Epoch  203  time  1.9567017555236816
Average time per epoch 0.925425594329834 (for 500 epochs)
Epoch  204  time  2.0369813442230225
Average time per epoch 0.92949955701828 (for 500 epochs)
Epoch  205  time  1.9570913314819336
Average time per epoch 0.9334137396812439 (for 500 epochs)
Epoch  206  time  1.9860167503356934
Average time per epoch 0.9373857731819153 (for 500 epochs)
Epoch  207  time  1.961681604385376
Average time per epoch 0.941309136390686 (for 500 epochs)
Epoch  208  time  2.0665416717529297
Average time per epoch 0.9454422197341918 (for 500 epochs)
Epoch  209  time  2.002258777618408
Average time per epoch 0.9494467372894287 (for 500 epochs)
Epoch  210  time  1.9689264297485352
Epoch  210  loss  1.7187680036816704 correct 48
Average time per epoch 0.9533845901489257 (for 500 epochs)
Epoch  211  time  2.138935089111328
Average time per epoch 0.9576624603271484 (for 500 epochs)
Epoch  212  time  1.9706802368164062
Average time per epoch 0.9616038208007812 (for 500 epochs)
Epoch  213  time  1.9975693225860596
Average time per epoch 0.9655989594459534 (for 500 epochs)
Epoch  214  time  1.9815728664398193
Average time per epoch 0.969562105178833 (for 500 epochs)
Epoch  215  time  2.0556111335754395
Average time per epoch 0.9736733274459839 (for 500 epochs)
Epoch  216  time  1.9977948665618896
Average time per epoch 0.9776689171791076 (for 500 epochs)
Epoch  217  time  1.98997163772583
Average time per epoch 0.9816488604545593 (for 500 epochs)
Epoch  218  time  2.037667751312256
Average time per epoch 0.9857241959571839 (for 500 epochs)
Epoch  219  time  1.9972164630889893
Average time per epoch 0.9897186288833618 (for 500 epochs)
Epoch  220  time  1.9918279647827148
Epoch  220  loss  1.29175856075797 correct 49
Average time per epoch 0.9937022848129272 (for 500 epochs)
Epoch  221  time  2.0135161876678467
Average time per epoch 0.997729317188263 (for 500 epochs)
Epoch  222  time  2.0248100757598877
Average time per epoch 1.0017789373397827 (for 500 epochs)
Epoch  223  time  2.801485776901245
Average time per epoch 1.0073819088935851 (for 500 epochs)
Epoch  224  time  3.430980682373047
Average time per epoch 1.0142438702583314 (for 500 epochs)
Epoch  225  time  3.5563578605651855
Average time per epoch 1.0213565859794618 (for 500 epochs)
Epoch  226  time  2.970452070236206
Average time per epoch 1.027297490119934 (for 500 epochs)
Epoch  227  time  1.9993455410003662
Average time per epoch 1.0312961812019348 (for 500 epochs)
Epoch  228  time  2.030667781829834
Average time per epoch 1.0353575167655944 (for 500 epochs)
Epoch  229  time  1.9942584037780762
Average time per epoch 1.0393460335731506 (for 500 epochs)
Epoch  230  time  1.9882354736328125
Epoch  230  loss  1.5818748924110915 correct 46
Average time per epoch 1.0433225045204162 (for 500 epochs)
Epoch  231  time  1.9706358909606934
Average time per epoch 1.0472637763023376 (for 500 epochs)
Epoch  232  time  2.0490002632141113
Average time per epoch 1.051361776828766 (for 500 epochs)
Epoch  233  time  1.9798438549041748
Average time per epoch 1.0553214645385742 (for 500 epochs)
Epoch  234  time  2.0125980377197266
Average time per epoch 1.0593466606140136 (for 500 epochs)
Epoch  235  time  2.0721676349639893
Average time per epoch 1.0634909958839416 (for 500 epochs)
Epoch  236  time  1.99420166015625
Average time per epoch 1.0674793992042542 (for 500 epochs)
Epoch  237  time  1.9812397956848145
Average time per epoch 1.0714418787956237 (for 500 epochs)
Epoch  238  time  2.0085856914520264
Average time per epoch 1.075459050178528 (for 500 epochs)
Epoch  239  time  2.0677385330200195
Average time per epoch 1.079594527244568 (for 500 epochs)
Epoch  240  time  1.9686479568481445
Epoch  240  loss  0.1855513267229256 correct 48
Average time per epoch 1.0835318231582642 (for 500 epochs)
Epoch  241  time  1.9998013973236084
Average time per epoch 1.0875314259529114 (for 500 epochs)
Epoch  242  time  2.0564165115356445
Average time per epoch 1.0916442589759827 (for 500 epochs)
Epoch  243  time  1.963681936264038
Average time per epoch 1.0955716228485108 (for 500 epochs)
Epoch  244  time  1.9955527782440186
Average time per epoch 1.0995627284049987 (for 500 epochs)
Epoch  245  time  2.0336272716522217
Average time per epoch 1.1036299829483032 (for 500 epochs)
Epoch  246  time  1.9581074714660645
Average time per epoch 1.1075461978912353 (for 500 epochs)
Epoch  247  time  2.001380205154419
Average time per epoch 1.111548958301544 (for 500 epochs)
Epoch  248  time  1.9666359424591064
Average time per epoch 1.1154822301864624 (for 500 epochs)
Epoch  249  time  2.044628381729126
Average time per epoch 1.1195714869499207 (for 500 epochs)
Epoch  250  time  1.9744791984558105
Epoch  250  loss  0.3190442381485551 correct 44
Average time per epoch 1.1235204453468324 (for 500 epochs)
Epoch  251  time  1.993947982788086
Average time per epoch 1.1275083413124085 (for 500 epochs)
Epoch  252  time  2.0321121215820312
Average time per epoch 1.1315725655555726 (for 500 epochs)
Epoch  253  time  2.0272443294525146
Average time per epoch 1.1356270542144775 (for 500 epochs)
Epoch  254  time  2.0222809314727783
Average time per epoch 1.1396716160774232 (for 500 epochs)
Epoch  255  time  2.0491960048675537
Average time per epoch 1.1437700080871582 (for 500 epochs)
Epoch  256  time  1.998605728149414
Average time per epoch 1.147767219543457 (for 500 epochs)
Epoch  257  time  2.8176779747009277
Average time per epoch 1.153402575492859 (for 500 epochs)
Epoch  258  time  3.4253947734832764
Average time per epoch 1.1602533650398255 (for 500 epochs)
Epoch  259  time  3.5289885997772217
Average time per epoch 1.16731134223938 (for 500 epochs)
Epoch  260  time  3.0289406776428223
Epoch  260  loss  2.7129192218951994 correct 48
Average time per epoch 1.1733692235946656 (for 500 epochs)
Epoch  261  time  1.968595266342163
Average time per epoch 1.17730641412735 (for 500 epochs)
Epoch  262  time  2.02862548828125
Average time per epoch 1.1813636651039123 (for 500 epochs)
Epoch  263  time  1.9918148517608643
Average time per epoch 1.1853472948074342 (for 500 epochs)
Epoch  264  time  1.9580483436584473
Average time per epoch 1.189263391494751 (for 500 epochs)
Epoch  265  time  1.9710943698883057
Average time per epoch 1.1932055802345276 (for 500 epochs)
Epoch  266  time  2.060232400894165
Average time per epoch 1.197326045036316 (for 500 epochs)
Epoch  267  time  1.9857556819915771
Average time per epoch 1.201297556400299 (for 500 epochs)
Epoch  268  time  1.9795825481414795
Average time per epoch 1.2052567214965821 (for 500 epochs)
Epoch  269  time  2.038426637649536
Average time per epoch 1.2093335747718812 (for 500 epochs)
Epoch  270  time  1.9860692024230957
Epoch  270  loss  1.139502191064107 correct 49
Average time per epoch 1.2133057131767273 (for 500 epochs)
Epoch  271  time  2.002546787261963
Average time per epoch 1.2173108067512513 (for 500 epochs)
Epoch  272  time  2.030768871307373
Average time per epoch 1.221372344493866 (for 500 epochs)
Epoch  273  time  1.9530665874481201
Average time per epoch 1.2252784776687622 (for 500 epochs)
Epoch  274  time  1.9793531894683838
Average time per epoch 1.229237184047699 (for 500 epochs)
Epoch  275  time  1.9775993824005127
Average time per epoch 1.2331923828125 (for 500 epochs)
Epoch  276  time  2.0355422496795654
Average time per epoch 1.2372634673118592 (for 500 epochs)
Epoch  277  time  1.9552180767059326
Average time per epoch 1.241173903465271 (for 500 epochs)
Epoch  278  time  1.9482638835906982
Average time per epoch 1.2450704312324523 (for 500 epochs)
Epoch  279  time  2.0787665843963623
Average time per epoch 1.2492279644012452 (for 500 epochs)
Epoch  280  time  3.101860761642456
Epoch  280  loss  0.6213664197311299 correct 48
Average time per epoch 1.25543168592453 (for 500 epochs)
Epoch  281  time  3.439345598220825
Average time per epoch 1.2623103771209716 (for 500 epochs)
Epoch  282  time  3.511826992034912
Average time per epoch 1.2693340311050414 (for 500 epochs)
Epoch  283  time  2.703242063522339
Average time per epoch 1.274740515232086 (for 500 epochs)
Epoch  284  time  1.9614362716674805
Average time per epoch 1.2786633877754212 (for 500 epochs)
Epoch  285  time  1.9729888439178467
Average time per epoch 1.2826093654632569 (for 500 epochs)
Epoch  286  time  2.0214014053344727
Average time per epoch 1.2866521682739258 (for 500 epochs)
Epoch  287  time  1.9819097518920898
Average time per epoch 1.29061598777771 (for 500 epochs)
Epoch  288  time  1.9788031578063965
Average time per epoch 1.2945735940933227 (for 500 epochs)
Epoch  289  time  3.170969009399414
Average time per epoch 1.3009155321121215 (for 500 epochs)
Epoch  290  time  3.4697558879852295
Epoch  290  loss  1.8199655337576186 correct 49
Average time per epoch 1.307855043888092 (for 500 epochs)
Epoch  291  time  3.41288423538208
Average time per epoch 1.3146808123588563 (for 500 epochs)
Epoch  292  time  2.7267568111419678
Average time per epoch 1.3201343259811402 (for 500 epochs)
Epoch  293  time  2.0529282093048096
Average time per epoch 1.3242401823997498 (for 500 epochs)
Epoch  294  time  2.015126943588257
Average time per epoch 1.3282704362869262 (for 500 epochs)
Epoch  295  time  1.9615881443023682
Average time per epoch 1.332193612575531 (for 500 epochs)
Epoch  296  time  2.047638177871704
Average time per epoch 1.3362888889312745 (for 500 epochs)
Epoch  297  time  1.9855670928955078
Average time per epoch 1.3402600231170654 (for 500 epochs)
Epoch  298  time  1.963484764099121
Average time per epoch 1.3441869926452636 (for 500 epochs)
Epoch  299  time  2.028196334838867
Average time per epoch 1.3482433853149414 (for 500 epochs)
Epoch  300  time  1.9704744815826416
Epoch  300  loss  0.39395722188615856 correct 49
Average time per epoch 1.3521843342781066 (for 500 epochs)
Epoch  301  time  1.979921817779541
Average time per epoch 1.3561441779136658 (for 500 epochs)
Epoch  302  time  1.9808166027069092
Average time per epoch 1.3601058111190796 (for 500 epochs)
Epoch  303  time  2.2045247554779053
Average time per epoch 1.3645148606300355 (for 500 epochs)
Epoch  304  time  3.24845027923584
Average time per epoch 1.371011761188507 (for 500 epochs)
Epoch  305  time  3.412541151046753
Average time per epoch 1.3778368434906005 (for 500 epochs)
Epoch  306  time  3.495196580886841
Average time per epoch 1.3848272366523742 (for 500 epochs)
Epoch  307  time  2.396479845046997
Average time per epoch 1.3896201963424684 (for 500 epochs)
Epoch  308  time  1.9609010219573975
Average time per epoch 1.393541998386383 (for 500 epochs)
Epoch  309  time  1.9505443572998047
Average time per epoch 1.3974430871009826 (for 500 epochs)
Epoch  310  time  2.0538716316223145
Epoch  310  loss  1.3412232013449765 correct 48
Average time per epoch 1.4015508303642272 (for 500 epochs)
Epoch  311  time  1.97713303565979
Average time per epoch 1.4055050964355469 (for 500 epochs)
Epoch  312  time  1.9541072845458984
Average time per epoch 1.4094133110046387 (for 500 epochs)
Epoch  313  time  2.0370752811431885
Average time per epoch 1.413487461566925 (for 500 epochs)
Epoch  314  time  1.9681828022003174
Average time per epoch 1.4174238271713258 (for 500 epochs)
Epoch  315  time  1.9817543029785156
Average time per epoch 1.4213873357772828 (for 500 epochs)
Epoch  316  time  2.0809326171875
Average time per epoch 1.4255492010116577 (for 500 epochs)
Epoch  317  time  1.9718644618988037
Average time per epoch 1.4294929299354553 (for 500 epochs)
Epoch  318  time  2.0252749919891357
Average time per epoch 1.4335434799194335 (for 500 epochs)
Epoch  319  time  2.1162753105163574
Average time per epoch 1.4377760305404663 (for 500 epochs)
Epoch  320  time  3.3578433990478516
Epoch  320  loss  1.2979057671223462 correct 50
Average time per epoch 1.444491717338562 (for 500 epochs)
Epoch  321  time  3.3861453533172607
Average time per epoch 1.4512640080451966 (for 500 epochs)
Epoch  322  time  3.3955953121185303
Average time per epoch 1.4580551986694337 (for 500 epochs)
Epoch  323  time  2.5110082626342773
Average time per epoch 1.4630772151947022 (for 500 epochs)
Epoch  324  time  1.956589698791504
Average time per epoch 1.466990394592285 (for 500 epochs)
Epoch  325  time  1.9962332248687744
Average time per epoch 1.4709828610420228 (for 500 epochs)
Epoch  326  time  2.0456931591033936
Average time per epoch 1.4750742473602294 (for 500 epochs)
Epoch  327  time  1.965052843093872
Average time per epoch 1.4790043530464172 (for 500 epochs)
Epoch  328  time  1.945906400680542
Average time per epoch 1.4828961658477784 (for 500 epochs)
Epoch  329  time  1.9729762077331543
Average time per epoch 1.4868421182632445 (for 500 epochs)
Epoch  330  time  2.031503915786743
Epoch  330  loss  1.3838600057124288 correct 49
Average time per epoch 1.490905126094818 (for 500 epochs)
Epoch  331  time  2.009561538696289
Average time per epoch 1.4949242491722108 (for 500 epochs)
Epoch  332  time  1.9823112487792969
Average time per epoch 1.4988888716697693 (for 500 epochs)
Epoch  333  time  2.0837998390197754
Average time per epoch 1.5030564713478087 (for 500 epochs)
Epoch  334  time  1.9978914260864258
Average time per epoch 1.5070522541999818 (for 500 epochs)
Epoch  335  time  1.9628477096557617
Average time per epoch 1.5109779496192932 (for 500 epochs)
Epoch  336  time  2.0051281452178955
Average time per epoch 1.514988205909729 (for 500 epochs)
Epoch  337  time  2.0193135738372803
Average time per epoch 1.5190268330574035 (for 500 epochs)
Epoch  338  time  1.9720346927642822
Average time per epoch 1.522970902442932 (for 500 epochs)
Epoch  339  time  1.946800947189331
Average time per epoch 1.5268645043373108 (for 500 epochs)
Epoch  340  time  2.0474584102630615
Epoch  340  loss  1.1404747390266907 correct 49
Average time per epoch 1.530959421157837 (for 500 epochs)
Epoch  341  time  1.989617109298706
Average time per epoch 1.5349386553764344 (for 500 epochs)
Epoch  342  time  1.958146333694458
Average time per epoch 1.5388549480438232 (for 500 epochs)
Epoch  343  time  2.041289806365967
Average time per epoch 1.5429375276565551 (for 500 epochs)
Epoch  344  time  2.005117177963257
Average time per epoch 1.5469477620124816 (for 500 epochs)
Epoch  345  time  1.9653916358947754
Average time per epoch 1.5508785452842713 (for 500 epochs)
Epoch  346  time  1.969980001449585
Average time per epoch 1.5548185052871704 (for 500 epochs)
Epoch  347  time  2.0240790843963623
Average time per epoch 1.5588666634559631 (for 500 epochs)
Epoch  348  time  1.9475901126861572
Average time per epoch 1.5627618436813355 (for 500 epochs)
Epoch  349  time  1.9991388320922852
Average time per epoch 1.56676012134552 (for 500 epochs)
Epoch  350  time  2.034975051879883
Epoch  350  loss  1.55876221002663 correct 49
Average time per epoch 1.5708300714492798 (for 500 epochs)
Epoch  351  time  1.9861760139465332
Average time per epoch 1.574802423477173 (for 500 epochs)
Epoch  352  time  1.940551519393921
Average time per epoch 1.5786835265159607 (for 500 epochs)
Epoch  353  time  1.9777674674987793
Average time per epoch 1.5826390614509582 (for 500 epochs)
Epoch  354  time  2.7248358726501465
Average time per epoch 1.5880887331962585 (for 500 epochs)
Epoch  355  time  3.4171981811523438
Average time per epoch 1.5949231295585633 (for 500 epochs)
Epoch  356  time  3.4081294536590576
Average time per epoch 1.6017393884658813 (for 500 epochs)
Epoch  357  time  3.2115869522094727
Average time per epoch 1.6081625623703002 (for 500 epochs)
Epoch  358  time  2.003507614135742
Average time per epoch 1.6121695775985718 (for 500 epochs)
Epoch  359  time  2.0059926509857178
Average time per epoch 1.6161815629005432 (for 500 epochs)
Epoch  360  time  2.040257453918457
Epoch  360  loss  0.342614988045286 correct 49
Average time per epoch 1.62026207780838 (for 500 epochs)
Epoch  361  time  1.979308843612671
Average time per epoch 1.6242206954956055 (for 500 epochs)
Epoch  362  time  2.042015790939331
Average time per epoch 1.6283047270774842 (for 500 epochs)
Epoch  363  time  1.979541301727295
Average time per epoch 1.6322638096809388 (for 500 epochs)
Epoch  364  time  2.076353073120117
Average time per epoch 1.636416515827179 (for 500 epochs)
Epoch  365  time  1.9411771297454834
Average time per epoch 1.6402988700866699 (for 500 epochs)
Epoch  366  time  1.9378433227539062
Average time per epoch 1.6441745567321777 (for 500 epochs)
Epoch  367  time  2.0184290409088135
Average time per epoch 1.6482114148139955 (for 500 epochs)
Epoch  368  time  1.952441692352295
Average time per epoch 1.6521162981986999 (for 500 epochs)
Epoch  369  time  1.9438996315002441
Average time per epoch 1.6560040974617005 (for 500 epochs)
Epoch  370  time  1.9694435596466064
Epoch  370  loss  2.6636533545949956 correct 46
Average time per epoch 1.6599429845809937 (for 500 epochs)
Epoch  371  time  2.0442492961883545
Average time per epoch 1.6640314831733705 (for 500 epochs)
Epoch  372  time  1.969055414199829
Average time per epoch 1.66796959400177 (for 500 epochs)
Epoch  373  time  1.9668433666229248
Average time per epoch 1.6719032807350158 (for 500 epochs)
Epoch  374  time  2.035855770111084
Average time per epoch 1.675974992275238 (for 500 epochs)
Epoch  375  time  1.975813627243042
Average time per epoch 1.6799266195297242 (for 500 epochs)
Epoch  376  time  1.9445643424987793
Average time per epoch 1.6838157482147216 (for 500 epochs)
Epoch  377  time  2.0783519744873047
Average time per epoch 1.6879724521636963 (for 500 epochs)
Epoch  378  time  1.959700107574463
Average time per epoch 1.6918918523788453 (for 500 epochs)
Epoch  379  time  1.9596424102783203
Average time per epoch 1.6958111371994018 (for 500 epochs)
Epoch  380  time  1.9629428386688232
Epoch  380  loss  0.7483945248490196 correct 48
Average time per epoch 1.6997370228767394 (for 500 epochs)
Epoch  381  time  2.0472705364227295
Average time per epoch 1.703831563949585 (for 500 epochs)
Epoch  382  time  2.024531841278076
Average time per epoch 1.7078806276321412 (for 500 epochs)
Epoch  383  time  1.960191249847412
Average time per epoch 1.711801010131836 (for 500 epochs)
Epoch  384  time  2.0490593910217285
Average time per epoch 1.7158991289138794 (for 500 epochs)
Epoch  385  time  2.0155093669891357
Average time per epoch 1.7199301476478577 (for 500 epochs)
Epoch  386  time  1.9822895526885986
Average time per epoch 1.723894726753235 (for 500 epochs)
Epoch  387  time  2.6066620349884033
Average time per epoch 1.7291080508232117 (for 500 epochs)
Epoch  388  time  3.4970476627349854
Average time per epoch 1.7361021461486816 (for 500 epochs)
Epoch  389  time  3.3996739387512207
Average time per epoch 1.742901494026184 (for 500 epochs)
Epoch  390  time  3.2229607105255127
Epoch  390  loss  0.9720079212434346 correct 46
Average time per epoch 1.749347415447235 (for 500 epochs)
Epoch  391  time  2.0438005924224854
Average time per epoch 1.75343501663208 (for 500 epochs)
Epoch  392  time  2.0298726558685303
Average time per epoch 1.757494761943817 (for 500 epochs)
Epoch  393  time  1.9497690200805664
Average time per epoch 1.7613942999839782 (for 500 epochs)
Epoch  394  time  2.008690357208252
Average time per epoch 1.7654116806983948 (for 500 epochs)
Epoch  395  time  1.9475455284118652
Average time per epoch 1.7693067717552184 (for 500 epochs)
Epoch  396  time  1.9667630195617676
Average time per epoch 1.773240297794342 (for 500 epochs)
Epoch  397  time  1.9664082527160645
Average time per epoch 1.7771731142997742 (for 500 epochs)
Epoch  398  time  2.0552937984466553
Average time per epoch 1.7812837018966674 (for 500 epochs)
Epoch  399  time  1.9940946102142334
Average time per epoch 1.785271891117096 (for 500 epochs)
Epoch  400  time  1.980100393295288
Epoch  400  loss  0.5284350075995874 correct 49
Average time per epoch 1.7892320919036866 (for 500 epochs)
Epoch  401  time  2.0918991565704346
Average time per epoch 1.7934158902168273 (for 500 epochs)
Epoch  402  time  2.0144383907318115
Average time per epoch 1.797444766998291 (for 500 epochs)
Epoch  403  time  1.9808883666992188
Average time per epoch 1.8014065437316895 (for 500 epochs)
Epoch  404  time  2.0303492546081543
Average time per epoch 1.8054672422409057 (for 500 epochs)
Epoch  405  time  1.984931468963623
Average time per epoch 1.809437105178833 (for 500 epochs)
Epoch  406  time  1.994457483291626
Average time per epoch 1.8134260201454162 (for 500 epochs)
Epoch  407  time  1.9611363410949707
Average time per epoch 1.8173482928276061 (for 500 epochs)
Epoch  408  time  2.0264346599578857
Average time per epoch 1.821401162147522 (for 500 epochs)
Epoch  409  time  1.9641451835632324
Average time per epoch 1.8253294525146484 (for 500 epochs)
Epoch  410  time  1.9548189640045166
Epoch  410  loss  0.04970141539810248 correct 47
Average time per epoch 1.8292390904426574 (for 500 epochs)
Epoch  411  time  2.052042245864868
Average time per epoch 1.8333431749343871 (for 500 epochs)
Epoch  412  time  1.9436988830566406
Average time per epoch 1.8372305727005005 (for 500 epochs)
Epoch  413  time  1.9699375629425049
Average time per epoch 1.8411704478263855 (for 500 epochs)
Epoch  414  time  3.005850076675415
Average time per epoch 1.8471821479797363 (for 500 epochs)
Epoch  415  time  3.5187976360321045
Average time per epoch 1.8542197432518006 (for 500 epochs)
Epoch  416  time  3.358182668685913
Average time per epoch 1.8609361085891725 (for 500 epochs)
Epoch  417  time  2.93479323387146
Average time per epoch 1.8668056950569152 (for 500 epochs)
Epoch  418  time  3.339185953140259
Average time per epoch 1.8734840669631958 (for 500 epochs)
Epoch  419  time  3.383667230606079
Average time per epoch 1.880251401424408 (for 500 epochs)
Epoch  420  time  3.3317716121673584
Epoch  420  loss  0.4164158429837464 correct 47
Average time per epoch 1.8869149446487428 (for 500 epochs)
Epoch  421  time  2.4651262760162354
Average time per epoch 1.891845197200775 (for 500 epochs)
Epoch  422  time  1.957381248474121
Average time per epoch 1.8957599596977235 (for 500 epochs)
Epoch  423  time  1.9831786155700684
Average time per epoch 1.8997263169288636 (for 500 epochs)
Epoch  424  time  2.0040135383605957
Average time per epoch 1.9037343440055847 (for 500 epochs)
Epoch  425  time  2.022279977798462
Average time per epoch 1.9077789039611817 (for 500 epochs)
Epoch  426  time  1.9489965438842773
Average time per epoch 1.9116768970489502 (for 500 epochs)
Epoch  427  time  2.0140061378479004
Average time per epoch 1.915704909324646 (for 500 epochs)
Epoch  428  time  2.091268301010132
Average time per epoch 1.9198874459266662 (for 500 epochs)
Epoch  429  time  1.953660011291504
Average time per epoch 1.9237947659492494 (for 500 epochs)
Epoch  430  time  1.945450782775879
Epoch  430  loss  0.2904258234199414 correct 48
Average time per epoch 1.927685667514801 (for 500 epochs)
Epoch  431  time  2.000694751739502
Average time per epoch 1.93168705701828 (for 500 epochs)
Epoch  432  time  2.024113655090332
Average time per epoch 1.9357352843284608 (for 500 epochs)
Epoch  433  time  1.9585695266723633
Average time per epoch 1.9396524233818053 (for 500 epochs)
Epoch  434  time  1.953303575515747
Average time per epoch 1.943559030532837 (for 500 epochs)
Epoch  435  time  2.033853769302368
Average time per epoch 1.9476267380714416 (for 500 epochs)
Epoch  436  time  1.9714033603668213
Average time per epoch 1.9515695447921753 (for 500 epochs)
Epoch  437  time  1.9622066020965576
Average time per epoch 1.9554939579963684 (for 500 epochs)
Epoch  438  time  2.060793161392212
Average time per epoch 1.9596155443191527 (for 500 epochs)
Epoch  439  time  1.9850435256958008
Average time per epoch 1.9635856313705444 (for 500 epochs)
Epoch  440  time  2.9826414585113525
Epoch  440  loss  0.4686488994651512 correct 50
Average time per epoch 1.9695509142875671 (for 500 epochs)
Epoch  441  time  3.4079627990722656
Average time per epoch 1.9763668398857117 (for 500 epochs)
Epoch  442  time  3.448131799697876
Average time per epoch 1.9832631034851074 (for 500 epochs)
Epoch  443  time  2.7431516647338867
Average time per epoch 1.9887494068145752 (for 500 epochs)
Epoch  444  time  2.039788007736206
Average time per epoch 1.9928289828300476 (for 500 epochs)
Epoch  445  time  2.0382003784179688
Average time per epoch 1.9969053835868835 (for 500 epochs)
Epoch  446  time  1.9485499858856201
Average time per epoch 2.0008024835586546 (for 500 epochs)
Epoch  447  time  1.952756643295288
Average time per epoch 2.0047079968452453 (for 500 epochs)
Epoch  448  time  2.246638536453247
Average time per epoch 2.0092012739181517 (for 500 epochs)
Epoch  449  time  3.1957859992980957
Average time per epoch 2.015592845916748 (for 500 epochs)
Epoch  450  time  3.380873680114746
Epoch  450  loss  0.23890006179130827 correct 50
Average time per epoch 2.0223545932769778 (for 500 epochs)
Epoch  451  time  3.3410305976867676
Average time per epoch 2.029036654472351 (for 500 epochs)
Epoch  452  time  2.4779624938964844
Average time per epoch 2.033992579460144 (for 500 epochs)
Epoch  453  time  1.9599251747131348
Average time per epoch 2.03791242980957 (for 500 epochs)
Epoch  454  time  1.9422099590301514
Average time per epoch 2.0417968497276306 (for 500 epochs)
Epoch  455  time  2.0015785694122314
Average time per epoch 2.0458000068664552 (for 500 epochs)
Epoch  456  time  1.9494874477386475
Average time per epoch 2.0496989817619324 (for 500 epochs)
Epoch  457  time  2.019315242767334
Average time per epoch 2.053737612247467 (for 500 epochs)
Epoch  458  time  1.9906647205352783
Average time per epoch 2.0577189416885378 (for 500 epochs)
Epoch  459  time  2.0228092670440674
Average time per epoch 2.0617645602226258 (for 500 epochs)
Epoch  460  time  1.96419095993042
Epoch  460  loss  0.3116158917186678 correct 49
Average time per epoch 2.0656929421424866 (for 500 epochs)
Epoch  461  time  1.955631971359253
Average time per epoch 2.069604206085205 (for 500 epochs)
Epoch  462  time  2.0284745693206787
Average time per epoch 2.0736611552238466 (for 500 epochs)
Epoch  463  time  1.9838368892669678
Average time per epoch 2.0776288290023803 (for 500 epochs)
Epoch  464  time  1.9856510162353516
Average time per epoch 2.081600131034851 (for 500 epochs)
Epoch  465  time  1.9475939273834229
Average time per epoch 2.0854953188896177 (for 500 epochs)
Epoch  466  time  2.0227696895599365
Average time per epoch 2.089540858268738 (for 500 epochs)
Epoch  467  time  2.0446746349334717
Average time per epoch 2.0936302075386046 (for 500 epochs)
Epoch  468  time  1.95548677444458
Average time per epoch 2.097541181087494 (for 500 epochs)
Epoch  469  time  2.0554051399230957
Average time per epoch 2.10165199136734 (for 500 epochs)
Epoch  470  time  1.9793906211853027
Epoch  470  loss  0.04368922558632647 correct 50
Average time per epoch 2.105610772609711 (for 500 epochs)
Epoch  471  time  1.9835338592529297
Average time per epoch 2.1095778403282166 (for 500 epochs)
Epoch  472  time  2.07331919670105
Average time per epoch 2.1137244787216187 (for 500 epochs)
Epoch  473  time  1.9746713638305664
Average time per epoch 2.1176738214492796 (for 500 epochs)
Epoch  474  time  1.98231840133667
Average time per epoch 2.121638458251953 (for 500 epochs)
Epoch  475  time  1.9480009078979492
Average time per epoch 2.125534460067749 (for 500 epochs)
Epoch  476  time  2.0021238327026367
Average time per epoch 2.1295387077331545 (for 500 epochs)
Epoch  477  time  1.9625647068023682
Average time per epoch 2.133463837146759 (for 500 epochs)
Epoch  478  time  1.937429666519165
Average time per epoch 2.1373386964797976 (for 500 epochs)
Epoch  479  time  2.063129425048828
Average time per epoch 2.1414649553298952 (for 500 epochs)
Epoch  480  time  2.0000863075256348
Epoch  480  loss  0.4509748203326583 correct 49
Average time per epoch 2.1454651279449464 (for 500 epochs)
Epoch  481  time  1.9517958164215088
Average time per epoch 2.1493687195777893 (for 500 epochs)
Epoch  482  time  2.0118720531463623
Average time per epoch 2.153392463684082 (for 500 epochs)
Epoch  483  time  2.1060092449188232
Average time per epoch 2.1576044821739195 (for 500 epochs)
Epoch  484  time  3.202467918395996
Average time per epoch 2.1640094180107114 (for 500 epochs)
Epoch  485  time  3.3424293994903564
Average time per epoch 2.170694276809692 (for 500 epochs)
Epoch  486  time  3.523256301879883
Average time per epoch 2.177740789413452 (for 500 epochs)
Epoch  487  time  2.453777313232422
Average time per epoch 2.182648344039917 (for 500 epochs)
Epoch  488  time  1.9287304878234863
Average time per epoch 2.186505805015564 (for 500 epochs)
Epoch  489  time  2.025283098220825
Average time per epoch 2.1905563712120055 (for 500 epochs)
Epoch  490  time  1.9371435642242432
Epoch  490  loss  0.07238983943837402 correct 49
Average time per epoch 2.1944306583404543 (for 500 epochs)
Epoch  491  time  1.9635593891143799
Average time per epoch 2.198357777118683 (for 500 epochs)
Epoch  492  time  1.957345962524414
Average time per epoch 2.2022724690437316 (for 500 epochs)
Epoch  493  time  2.027698040008545
Average time per epoch 2.2063278651237486 (for 500 epochs)
Epoch  494  time  1.9624204635620117
Average time per epoch 2.210252706050873 (for 500 epochs)
Epoch  495  time  1.9836289882659912
Average time per epoch 2.214219964027405 (for 500 epochs)
Epoch  496  time  2.014866590499878
Average time per epoch 2.2182496972084045 (for 500 epochs)
Epoch  497  time  1.947887897491455
Average time per epoch 2.2221454730033874 (for 500 epochs)
Epoch  498  time  1.9578044414520264
Average time per epoch 2.2260610818862916 (for 500 epochs)
Epoch  499  time  2.0234837532043457
Average time per epoch 2.2301080493927 (for 500 epochs)
```
# GPU Simple Large Dataset:
```
Epoch  0  loss  8.242400113758572 correct 37
Average time per epoch 0.010462114334106445 (for 500 epochs)
Epoch  1  time  2.0040645599365234
Average time per epoch 0.014470243453979492 (for 500 epochs)
Epoch  2  time  2.045010805130005
Average time per epoch 0.018560265064239502 (for 500 epochs)
Epoch  3  time  1.9846904277801514
Average time per epoch 0.022529645919799803 (for 500 epochs)
Epoch  4  time  1.9627559185028076
Average time per epoch 0.02645515775680542 (for 500 epochs)
Epoch  5  time  2.019319534301758
Average time per epoch 0.030493796825408936 (for 500 epochs)
Epoch  6  time  2.0303726196289062
Average time per epoch 0.034554542064666745 (for 500 epochs)
Epoch  7  time  1.9845695495605469
Average time per epoch 0.03852368116378784 (for 500 epochs)
Epoch  8  time  1.9928796291351318
Average time per epoch 0.042509440422058106 (for 500 epochs)
Epoch  9  time  2.0364937782287598
Average time per epoch 0.046582427978515625 (for 500 epochs)
Epoch  10  time  1.990002155303955
Epoch  10  loss  3.13595573589705 correct 48
Average time per epoch 0.050562432289123535 (for 500 epochs)
Epoch  11  time  2.043442964553833
Average time per epoch 0.0546493182182312 (for 500 epochs)
Epoch  12  time  2.0373568534851074
Average time per epoch 0.05872403192520142 (for 500 epochs)
Epoch  13  time  1.9869468212127686
Average time per epoch 0.06269792556762695 (for 500 epochs)
Epoch  14  time  1.9871728420257568
Average time per epoch 0.06667227125167846 (for 500 epochs)
Epoch  15  time  2.030345916748047
Average time per epoch 0.07073296308517456 (for 500 epochs)
Epoch  16  time  3.1996114253997803
Average time per epoch 0.07713218593597412 (for 500 epochs)
Epoch  17  time  3.3823747634887695
Average time per epoch 0.08389693546295166 (for 500 epochs)
Epoch  18  time  3.4353480339050293
Average time per epoch 0.09076763153076171 (for 500 epochs)
Epoch  19  time  2.9781291484832764
Average time per epoch 0.09672388982772827 (for 500 epochs)
Epoch  20  time  1.9953131675720215
Epoch  20  loss  1.04271121642354 correct 47
Average time per epoch 0.10071451616287232 (for 500 epochs)
Epoch  21  time  2.0097496509552
Average time per epoch 0.10473401546478271 (for 500 epochs)
Epoch  22  time  1.976203203201294
Average time per epoch 0.1086864218711853 (for 500 epochs)
Epoch  23  time  2.0463836193084717
Average time per epoch 0.11277918910980225 (for 500 epochs)
Epoch  24  time  1.9550340175628662
Average time per epoch 0.11668925714492798 (for 500 epochs)
Epoch  25  time  1.9308347702026367
Average time per epoch 0.12055092668533325 (for 500 epochs)
Epoch  26  time  2.059366464614868
Average time per epoch 0.12466965961456299 (for 500 epochs)
Epoch  27  time  1.9689464569091797
Average time per epoch 0.12860755252838135 (for 500 epochs)
Epoch  28  time  1.966303825378418
Average time per epoch 0.1325401601791382 (for 500 epochs)
Epoch  29  time  1.9734985828399658
Average time per epoch 0.13648715734481812 (for 500 epochs)
Epoch  30  time  1.9985392093658447
Epoch  30  loss  0.35410249874607314 correct 50
Average time per epoch 0.1404842357635498 (for 500 epochs)
Epoch  31  time  1.965831995010376
Average time per epoch 0.14441589975357055 (for 500 epochs)
Epoch  32  time  1.9558258056640625
Average time per epoch 0.14832755136489867 (for 500 epochs)
Epoch  33  time  2.004747152328491
Average time per epoch 0.15233704566955567 (for 500 epochs)
Epoch  34  time  2.0160579681396484
Average time per epoch 0.15636916160583497 (for 500 epochs)
Epoch  35  time  1.9627633094787598
Average time per epoch 0.16029468822479248 (for 500 epochs)
Epoch  36  time  1.930166244506836
Average time per epoch 0.16415502071380617 (for 500 epochs)
Epoch  37  time  2.022655725479126
Average time per epoch 0.1682003321647644 (for 500 epochs)
Epoch  38  time  1.9431958198547363
Average time per epoch 0.17208672380447387 (for 500 epochs)
Epoch  39  time  1.9510786533355713
Average time per epoch 0.17598888111114502 (for 500 epochs)
Epoch  40  time  2.0008161067962646
Epoch  40  loss  0.6561332187154872 correct 48
Average time per epoch 0.17999051332473756 (for 500 epochs)
Epoch  41  time  1.9338371753692627
Average time per epoch 0.18385818767547607 (for 500 epochs)
Epoch  42  time  1.971693754196167
Average time per epoch 0.18780157518386842 (for 500 epochs)
Epoch  43  time  1.9534120559692383
Average time per epoch 0.19170839929580688 (for 500 epochs)
Epoch  44  time  2.0429294109344482
Average time per epoch 0.19579425811767578 (for 500 epochs)
Epoch  45  time  1.921379566192627
Average time per epoch 0.19963701725006103 (for 500 epochs)
Epoch  46  time  1.9429190158843994
Average time per epoch 0.20352285528182984 (for 500 epochs)
Epoch  47  time  1.9937264919281006
Average time per epoch 0.20751030826568603 (for 500 epochs)
Epoch  48  time  1.9462273120880127
Average time per epoch 0.21140276288986207 (for 500 epochs)
Epoch  49  time  1.9821441173553467
Average time per epoch 0.21536705112457274 (for 500 epochs)
Epoch  50  time  2.2868287563323975
Epoch  50  loss  2.5430250985672678 correct 48
Average time per epoch 0.21994070863723755 (for 500 epochs)
Epoch  51  time  3.3566389083862305
Average time per epoch 0.22665398645401 (for 500 epochs)
Epoch  52  time  3.421259641647339
Average time per epoch 0.23349650573730468 (for 500 epochs)
Epoch  53  time  3.4109816551208496
Average time per epoch 0.2403184690475464 (for 500 epochs)
Epoch  54  time  2.4571361541748047
Average time per epoch 0.24523274135589598 (for 500 epochs)
Epoch  55  time  1.958768606185913
Average time per epoch 0.24915027856826782 (for 500 epochs)
Epoch  56  time  1.9871208667755127
Average time per epoch 0.25312452030181887 (for 500 epochs)
Epoch  57  time  1.9499235153198242
Average time per epoch 0.2570243673324585 (for 500 epochs)
Epoch  58  time  2.0340843200683594
Average time per epoch 0.26109253597259524 (for 500 epochs)
Epoch  59  time  1.9934029579162598
Average time per epoch 0.26507934188842774 (for 500 epochs)
Epoch  60  time  1.988814115524292
Epoch  60  loss  3.0727840587362363 correct 48
Average time per epoch 0.2690569701194763 (for 500 epochs)
Epoch  61  time  2.0647575855255127
Average time per epoch 0.27318648529052736 (for 500 epochs)
Epoch  62  time  2.001627206802368
Average time per epoch 0.2771897397041321 (for 500 epochs)
Epoch  63  time  1.9430451393127441
Average time per epoch 0.28107582998275754 (for 500 epochs)
Epoch  64  time  1.94893217086792
Average time per epoch 0.28497369432449343 (for 500 epochs)
Epoch  65  time  1.9889910221099854
Average time per epoch 0.2889516763687134 (for 500 epochs)
Epoch  66  time  1.9651408195495605
Average time per epoch 0.2928819580078125 (for 500 epochs)
Epoch  67  time  1.9776077270507812
Average time per epoch 0.2968371734619141 (for 500 epochs)
Epoch  68  time  2.009570360183716
Average time per epoch 0.3008563141822815 (for 500 epochs)
Epoch  69  time  1.9519212245941162
Average time per epoch 0.3047601566314697 (for 500 epochs)
Epoch  70  time  1.939171314239502
Epoch  70  loss  3.6015741072965537 correct 46
Average time per epoch 0.30863849925994874 (for 500 epochs)
Epoch  71  time  1.9508240222930908
Average time per epoch 0.3125401473045349 (for 500 epochs)
Epoch  72  time  2.0222864151000977
Average time per epoch 0.3165847201347351 (for 500 epochs)
Epoch  73  time  1.939441204071045
Average time per epoch 0.3204636025428772 (for 500 epochs)
Epoch  74  time  1.964935541152954
Average time per epoch 0.3243934736251831 (for 500 epochs)
Epoch  75  time  1.9952309131622314
Average time per epoch 0.32838393545150757 (for 500 epochs)
Epoch  76  time  1.947906732559204
Average time per epoch 0.332279748916626 (for 500 epochs)
Epoch  77  time  2.00241756439209
Average time per epoch 0.3362845840454102 (for 500 epochs)
Epoch  78  time  1.993138313293457
Average time per epoch 0.3402708606719971 (for 500 epochs)
Epoch  79  time  2.0630149841308594
Average time per epoch 0.3443968906402588 (for 500 epochs)
Epoch  80  time  1.9474012851715088
Epoch  80  loss  1.0332122058305144 correct 48
Average time per epoch 0.3482916932106018 (for 500 epochs)
Epoch  81  time  1.9512181282043457
Average time per epoch 0.3521941294670105 (for 500 epochs)
Epoch  82  time  2.0314488410949707
Average time per epoch 0.35625702714920043 (for 500 epochs)
Epoch  83  time  2.280346393585205
Average time per epoch 0.36081771993637085 (for 500 epochs)
Epoch  84  time  3.222113609313965
Average time per epoch 0.36726194715499877 (for 500 epochs)
Epoch  85  time  3.376394748687744
Average time per epoch 0.3740147366523743 (for 500 epochs)
Epoch  86  time  3.5080864429473877
Average time per epoch 0.38103090953826907 (for 500 epochs)
Epoch  87  time  2.534980535507202
Average time per epoch 0.38610087060928344 (for 500 epochs)
Epoch  88  time  1.9768726825714111
Average time per epoch 0.39005461597442626 (for 500 epochs)
Epoch  89  time  2.0483763217926025
Average time per epoch 0.3941513686180115 (for 500 epochs)
Epoch  90  time  2.025120258331299
Epoch  90  loss  3.756584322832587 correct 44
Average time per epoch 0.39820160913467406 (for 500 epochs)
Epoch  91  time  1.9265375137329102
Average time per epoch 0.4020546841621399 (for 500 epochs)
Epoch  92  time  1.933760643005371
Average time per epoch 0.4059222054481506 (for 500 epochs)
Epoch  93  time  2.0123345851898193
Average time per epoch 0.4099468746185303 (for 500 epochs)
Epoch  94  time  1.9609525203704834
Average time per epoch 0.4138687796592712 (for 500 epochs)
Epoch  95  time  1.9567313194274902
Average time per epoch 0.4177822422981262 (for 500 epochs)
Epoch  96  time  2.0280814170837402
Average time per epoch 0.4218384051322937 (for 500 epochs)
Epoch  97  time  1.965270757675171
Average time per epoch 0.42576894664764403 (for 500 epochs)
Epoch  98  time  1.9660732746124268
Average time per epoch 0.4297010931968689 (for 500 epochs)
Epoch  99  time  1.9897618293762207
Average time per epoch 0.4336806168556213 (for 500 epochs)
Epoch  100  time  2.0530240535736084
Epoch  100  loss  0.11283661332620988 correct 48
Average time per epoch 0.4377866649627686 (for 500 epochs)
Epoch  101  time  2.0037193298339844
Average time per epoch 0.44179410362243654 (for 500 epochs)
Epoch  102  time  3.0102598667144775
Average time per epoch 0.44781462335586547 (for 500 epochs)
Epoch  103  time  3.4904356002807617
Average time per epoch 0.454795494556427 (for 500 epochs)
Epoch  104  time  3.5230331420898438
Average time per epoch 0.4618415608406067 (for 500 epochs)
Epoch  105  time  2.9904024600982666
Average time per epoch 0.4678223657608032 (for 500 epochs)
Epoch  106  time  1.950641393661499
Average time per epoch 0.4717236485481262 (for 500 epochs)
Epoch  107  time  2.0379700660705566
Average time per epoch 0.47579958868026734 (for 500 epochs)
Epoch  108  time  1.9467344284057617
Average time per epoch 0.47969305753707886 (for 500 epochs)
Epoch  109  time  1.9712750911712646
Average time per epoch 0.4836356077194214 (for 500 epochs)
Epoch  110  time  2.034750461578369
Epoch  110  loss  1.3981169030124787 correct 50
Average time per epoch 0.4877051086425781 (for 500 epochs)
Epoch  111  time  1.9909870624542236
Average time per epoch 0.49168708276748657 (for 500 epochs)
Epoch  112  time  1.9453012943267822
Average time per epoch 0.49557768535614016 (for 500 epochs)
Epoch  113  time  1.9654016494750977
Average time per epoch 0.49950848865509034 (for 500 epochs)
Epoch  114  time  2.0145182609558105
Average time per epoch 0.5035375251770019 (for 500 epochs)
Epoch  115  time  2.042116641998291
Average time per epoch 0.5076217584609986 (for 500 epochs)
Epoch  116  time  3.103088140487671
Average time per epoch 0.5138279347419739 (for 500 epochs)
Epoch  117  time  3.5675573348999023
Average time per epoch 0.5209630494117736 (for 500 epochs)
Epoch  118  time  3.3808257579803467
Average time per epoch 0.5277247009277344 (for 500 epochs)
Epoch  119  time  2.847321033477783
Average time per epoch 0.53341934299469 (for 500 epochs)
Epoch  120  time  1.9582023620605469
Epoch  120  loss  0.2954508814462349 correct 50
Average time per epoch 0.537335747718811 (for 500 epochs)
Epoch  121  time  2.0276076793670654
Average time per epoch 0.5413909630775452 (for 500 epochs)
Epoch  122  time  2.343482255935669
Average time per epoch 0.5460779275894165 (for 500 epochs)
Epoch  123  time  3.186689853668213
Average time per epoch 0.552451307296753 (for 500 epochs)
Epoch  124  time  3.496796131134033
Average time per epoch 0.559444899559021 (for 500 epochs)
Epoch  125  time  3.350215196609497
Average time per epoch 0.56614532995224 (for 500 epochs)
Epoch  126  time  2.4510574340820312
Average time per epoch 0.571047444820404 (for 500 epochs)
Epoch  127  time  1.9563992023468018
Average time per epoch 0.5749602432250976 (for 500 epochs)
Epoch  128  time  2.045945405960083
Average time per epoch 0.5790521340370178 (for 500 epochs)
Epoch  129  time  1.974468469619751
Average time per epoch 0.5830010709762573 (for 500 epochs)
Epoch  130  time  1.9871933460235596
Epoch  130  loss  1.0336145569814197 correct 48
Average time per epoch 0.5869754576683044 (for 500 epochs)
Epoch  131  time  2.0423104763031006
Average time per epoch 0.5910600786209106 (for 500 epochs)
Epoch  132  time  1.987086534500122
Average time per epoch 0.5950342516899109 (for 500 epochs)
Epoch  133  time  1.9634661674499512
Average time per epoch 0.5989611840248108 (for 500 epochs)
Epoch  134  time  1.9416770935058594
Average time per epoch 0.6028445382118225 (for 500 epochs)
Epoch  135  time  2.037000894546509
Average time per epoch 0.6069185400009155 (for 500 epochs)
Epoch  136  time  1.9412071704864502
Average time per epoch 0.6108009543418884 (for 500 epochs)
Epoch  137  time  1.9736108779907227
Average time per epoch 0.6147481760978699 (for 500 epochs)
Epoch  138  time  2.0244383811950684
Average time per epoch 0.61879705286026 (for 500 epochs)
Epoch  139  time  1.9584732055664062
Average time per epoch 0.6227139992713928 (for 500 epochs)
Epoch  140  time  1.9307951927185059
Epoch  140  loss  0.8866694847853214 correct 50
Average time per epoch 0.6265755896568298 (for 500 epochs)
Epoch  141  time  1.925518274307251
Average time per epoch 0.6304266262054443 (for 500 epochs)
Epoch  142  time  2.0085837841033936
Average time per epoch 0.6344437937736511 (for 500 epochs)
Epoch  143  time  1.9156060218811035
Average time per epoch 0.6382750058174134 (for 500 epochs)
Epoch  144  time  2.0106632709503174
Average time per epoch 0.6422963323593139 (for 500 epochs)
Epoch  145  time  2.0351924896240234
Average time per epoch 0.646366717338562 (for 500 epochs)
Epoch  146  time  1.9298553466796875
Average time per epoch 0.6502264280319214 (for 500 epochs)
Epoch  147  time  2.557691812515259
Average time per epoch 0.6553418116569519 (for 500 epochs)
Epoch  148  time  3.270665168762207
Average time per epoch 0.6618831419944763 (for 500 epochs)
Epoch  149  time  3.433027982711792
Average time per epoch 0.6687491979599 (for 500 epochs)
Epoch  150  time  3.360431432723999
Epoch  150  loss  1.3206447470907914 correct 49
Average time per epoch 0.6754700608253479 (for 500 epochs)
Epoch  151  time  2.13417649269104
Average time per epoch 0.67973841381073 (for 500 epochs)
Epoch  152  time  2.0196123123168945
Average time per epoch 0.6837776384353638 (for 500 epochs)
Epoch  153  time  1.9513275623321533
Average time per epoch 0.687680293560028 (for 500 epochs)
Epoch  154  time  1.9812521934509277
Average time per epoch 0.69164279794693 (for 500 epochs)
Epoch  155  time  1.9714689254760742
Average time per epoch 0.6955857357978821 (for 500 epochs)
Epoch  156  time  2.074629545211792
Average time per epoch 0.6997349948883057 (for 500 epochs)
Epoch  157  time  2.034444808959961
Average time per epoch 0.7038038845062256 (for 500 epochs)
Epoch  158  time  1.9409472942352295
Average time per epoch 0.7076857790946961 (for 500 epochs)
Epoch  159  time  2.0192625522613525
Average time per epoch 0.7117243041992187 (for 500 epochs)
Epoch  160  time  1.9499249458312988
Epoch  160  loss  0.5714812833020527 correct 49
Average time per epoch 0.7156241540908813 (for 500 epochs)
Epoch  161  time  1.9353411197662354
Average time per epoch 0.7194948363304138 (for 500 epochs)
Epoch  162  time  1.9666247367858887
Average time per epoch 0.7234280858039855 (for 500 epochs)
Epoch  163  time  2.0199663639068604
Average time per epoch 0.7274680185317993 (for 500 epochs)
Epoch  164  time  1.9461872577667236
Average time per epoch 0.7313603930473328 (for 500 epochs)
Epoch  165  time  1.9532604217529297
Average time per epoch 0.7352669138908386 (for 500 epochs)
Epoch  166  time  2.033421516418457
Average time per epoch 0.7393337569236755 (for 500 epochs)
Epoch  167  time  1.9797110557556152
Average time per epoch 0.7432931790351868 (for 500 epochs)
Epoch  168  time  1.9851644039154053
Average time per epoch 0.7472635078430175 (for 500 epochs)
Epoch  169  time  1.9658620357513428
Average time per epoch 0.7511952319145203 (for 500 epochs)
Epoch  170  time  2.0324175357818604
Epoch  170  loss  0.3390339921987581 correct 50
Average time per epoch 0.7552600669860839 (for 500 epochs)
Epoch  171  time  1.9391670227050781
Average time per epoch 0.7591384010314941 (for 500 epochs)
Epoch  172  time  1.9884893894195557
Average time per epoch 0.7631153798103333 (for 500 epochs)
Epoch  173  time  2.008043050765991
Average time per epoch 0.7671314659118652 (for 500 epochs)
Epoch  174  time  1.9463980197906494
Average time per epoch 0.7710242619514466 (for 500 epochs)
Epoch  175  time  1.9389870166778564
Average time per epoch 0.7749022359848022 (for 500 epochs)
Epoch  176  time  1.9615869522094727
Average time per epoch 0.7788254098892212 (for 500 epochs)
Epoch  177  time  1.9789934158325195
Average time per epoch 0.7827833967208863 (for 500 epochs)
Epoch  178  time  1.9141123294830322
Average time per epoch 0.7866116213798523 (for 500 epochs)
Epoch  179  time  1.9429609775543213
Average time per epoch 0.7904975433349609 (for 500 epochs)
Epoch  180  time  2.0125839710235596
Epoch  180  loss  1.0199649114407303 correct 50
Average time per epoch 0.794522711277008 (for 500 epochs)
Epoch  181  time  1.9575424194335938
Average time per epoch 0.7984377961158753 (for 500 epochs)
Epoch  182  time  3.047471046447754
Average time per epoch 0.8045327382087708 (for 500 epochs)
Epoch  183  time  3.3732941150665283
Average time per epoch 0.8112793264389038 (for 500 epochs)
Epoch  184  time  3.4693167209625244
Average time per epoch 0.8182179598808289 (for 500 epochs)
Epoch  185  time  3.039405107498169
Average time per epoch 0.8242967700958252 (for 500 epochs)
Epoch  186  time  1.984769344329834
Average time per epoch 0.8282663087844848 (for 500 epochs)
Epoch  187  time  2.016324043273926
Average time per epoch 0.8322989568710327 (for 500 epochs)
Epoch  188  time  1.9354164600372314
Average time per epoch 0.8361697897911072 (for 500 epochs)
Epoch  189  time  1.931837797164917
Average time per epoch 0.840033465385437 (for 500 epochs)
Epoch  190  time  1.9416077136993408
Epoch  190  loss  0.040356170854463594 correct 49
Average time per epoch 0.8439166808128357 (for 500 epochs)
Epoch  191  time  2.003500461578369
Average time per epoch 0.8479236817359924 (for 500 epochs)
Epoch  192  time  1.9367146492004395
Average time per epoch 0.8517971110343933 (for 500 epochs)
Epoch  193  time  1.9418559074401855
Average time per epoch 0.8556808228492737 (for 500 epochs)
Epoch  194  time  1.9295799732208252
Average time per epoch 0.8595399827957153 (for 500 epochs)
Epoch  195  time  2.0010876655578613
Average time per epoch 0.8635421581268311 (for 500 epochs)
Epoch  196  time  1.9293701648712158
Average time per epoch 0.8674008984565735 (for 500 epochs)
Epoch  197  time  1.9189984798431396
Average time per epoch 0.8712388954162598 (for 500 epochs)
Epoch  198  time  2.026182174682617
Average time per epoch 0.875291259765625 (for 500 epochs)
Epoch  199  time  1.950287103652954
Average time per epoch 0.8791918339729309 (for 500 epochs)
Epoch  200  time  1.9061212539672852
Epoch  200  loss  0.13300425123831502 correct 47
Average time per epoch 0.8830040764808654 (for 500 epochs)
Epoch  201  time  1.9815263748168945
Average time per epoch 0.8869671292304993 (for 500 epochs)
Epoch  202  time  2.01904559135437
Average time per epoch 0.891005220413208 (for 500 epochs)
Epoch  203  time  1.9262804985046387
Average time per epoch 0.8948577814102173 (for 500 epochs)
Epoch  204  time  1.9604566097259521
Average time per epoch 0.8987786946296692 (for 500 epochs)
Epoch  205  time  2.0275866985321045
Average time per epoch 0.9028338680267334 (for 500 epochs)
Epoch  206  time  1.9519336223602295
Average time per epoch 0.9067377352714538 (for 500 epochs)
Epoch  207  time  1.956758737564087
Average time per epoch 0.910651252746582 (for 500 epochs)
Epoch  208  time  1.9576923847198486
Average time per epoch 0.9145666375160217 (for 500 epochs)
Epoch  209  time  2.0161638259887695
Average time per epoch 0.9185989651679992 (for 500 epochs)
Epoch  210  time  1.9334936141967773
Epoch  210  loss  0.21478144672429467 correct 50
Average time per epoch 0.9224659523963928 (for 500 epochs)
Epoch  211  time  1.9260542392730713
Average time per epoch 0.926318060874939 (for 500 epochs)
Epoch  212  time  2.0088229179382324
Average time per epoch 0.9303357067108154 (for 500 epochs)
Epoch  213  time  1.938542127609253
Average time per epoch 0.9342127909660339 (for 500 epochs)
Epoch  214  time  1.932363510131836
Average time per epoch 0.9380775179862976 (for 500 epochs)
Epoch  215  time  1.9958267211914062
Average time per epoch 0.9420691714286804 (for 500 epochs)
Epoch  216  time  2.252687931060791
Average time per epoch 0.946574547290802 (for 500 epochs)
Epoch  217  time  3.1986989974975586
Average time per epoch 0.9529719452857971 (for 500 epochs)
Epoch  218  time  3.363104820251465
Average time per epoch 0.9596981549263001 (for 500 epochs)
Epoch  219  time  3.4463257789611816
Average time per epoch 0.9665908064842225 (for 500 epochs)
Epoch  220  time  2.7297966480255127
Epoch  220  loss  0.7670340055821853 correct 49
Average time per epoch 0.9720503997802734 (for 500 epochs)
Epoch  221  time  1.9363057613372803
Average time per epoch 0.975923011302948 (for 500 epochs)
Epoch  222  time  1.9493968486785889
Average time per epoch 0.9798218050003051 (for 500 epochs)
Epoch  223  time  2.0405218601226807
Average time per epoch 0.9839028487205506 (for 500 epochs)
Epoch  224  time  1.9422225952148438
Average time per epoch 0.9877872939109802 (for 500 epochs)
Epoch  225  time  1.9501893520355225
Average time per epoch 0.9916876726150513 (for 500 epochs)
Epoch  226  time  2.015066146850586
Average time per epoch 0.9957178049087524 (for 500 epochs)
Epoch  227  time  1.9365429878234863
Average time per epoch 0.9995908908843995 (for 500 epochs)
Epoch  228  time  1.928234338760376
Average time per epoch 1.0034473595619202 (for 500 epochs)
Epoch  229  time  1.935516357421875
Average time per epoch 1.007318392276764 (for 500 epochs)
Epoch  230  time  2.07593035697937
Epoch  230  loss  0.2062608754153656 correct 47
Average time per epoch 1.0114702529907227 (for 500 epochs)
Epoch  231  time  1.931746006011963
Average time per epoch 1.0153337450027466 (for 500 epochs)
Epoch  232  time  1.9615495204925537
Average time per epoch 1.0192568440437317 (for 500 epochs)
Epoch  233  time  1.9890973567962646
Average time per epoch 1.0232350387573241 (for 500 epochs)
Epoch  234  time  1.926727056503296
Average time per epoch 1.0270884928703308 (for 500 epochs)
Epoch  235  time  1.9774274826049805
Average time per epoch 1.0310433478355407 (for 500 epochs)
Epoch  236  time  1.9438352584838867
Average time per epoch 1.0349310183525084 (for 500 epochs)
Epoch  237  time  2.0871894359588623
Average time per epoch 1.0391053972244262 (for 500 epochs)
Epoch  238  time  2.5760138034820557
Average time per epoch 1.0442574248313903 (for 500 epochs)
Epoch  239  time  3.2873594760894775
Average time per epoch 1.0508321437835693 (for 500 epochs)
Epoch  240  time  3.519549608230591
Epoch  240  loss  0.7997556418541141 correct 49
Average time per epoch 1.0578712430000305 (for 500 epochs)
Epoch  241  time  3.4047188758850098
Average time per epoch 1.0646806807518006 (for 500 epochs)
Epoch  242  time  2.0687973499298096
Average time per epoch 1.0688182754516602 (for 500 epochs)
Epoch  243  time  1.9839158058166504
Average time per epoch 1.0727861070632934 (for 500 epochs)
Epoch  244  time  2.0953562259674072
Average time per epoch 1.0769768195152283 (for 500 epochs)
Epoch  245  time  2.0042498111724854
Average time per epoch 1.0809853191375733 (for 500 epochs)
Epoch  246  time  2.0110979080200195
Average time per epoch 1.0850075149536134 (for 500 epochs)
Epoch  247  time  1.9670474529266357
Average time per epoch 1.0889416098594666 (for 500 epochs)
Epoch  248  time  2.5313456058502197
Average time per epoch 1.094004301071167 (for 500 epochs)
Epoch  249  time  3.2477710247039795
Average time per epoch 1.100499843120575 (for 500 epochs)
Epoch  250  time  3.391341209411621
Epoch  250  loss  2.89616412559286 correct 45
Average time per epoch 1.107282525539398 (for 500 epochs)
Epoch  251  time  3.455632209777832
Average time per epoch 1.1141937899589538 (for 500 epochs)
Epoch  252  time  2.253993034362793
Average time per epoch 1.1187017760276794 (for 500 epochs)
Epoch  253  time  1.941042184829712
Average time per epoch 1.1225838603973388 (for 500 epochs)
Epoch  254  time  1.9581804275512695
Average time per epoch 1.1265002212524413 (for 500 epochs)
Epoch  255  time  2.0596680641174316
Average time per epoch 1.1306195573806763 (for 500 epochs)
Epoch  256  time  1.9326086044311523
Average time per epoch 1.1344847745895386 (for 500 epochs)
Epoch  257  time  1.959526777267456
Average time per epoch 1.1384038281440736 (for 500 epochs)
Epoch  258  time  1.994001865386963
Average time per epoch 1.1423918318748474 (for 500 epochs)
Epoch  259  time  1.9197840690612793
Average time per epoch 1.14623140001297 (for 500 epochs)
Epoch  260  time  2.174471378326416
Epoch  260  loss  0.23632402625009444 correct 50
Average time per epoch 1.1505803427696228 (for 500 epochs)
Epoch  261  time  3.1409032344818115
Average time per epoch 1.1568621492385864 (for 500 epochs)
Epoch  262  time  3.4473440647125244
Average time per epoch 1.1637568373680114 (for 500 epochs)
Epoch  263  time  3.396063804626465
Average time per epoch 1.1705489649772645 (for 500 epochs)
Epoch  264  time  2.6476242542266846
Average time per epoch 1.1758442134857179 (for 500 epochs)
Epoch  265  time  2.0181920528411865
Average time per epoch 1.1798805975914002 (for 500 epochs)
Epoch  266  time  1.9711170196533203
Average time per epoch 1.183822831630707 (for 500 epochs)
Epoch  267  time  1.9679203033447266
Average time per epoch 1.1877586722373963 (for 500 epochs)
Epoch  268  time  1.9890024662017822
Average time per epoch 1.1917366771697997 (for 500 epochs)
Epoch  269  time  2.0071632862091064
Average time per epoch 1.195751003742218 (for 500 epochs)
Epoch  270  time  1.997471570968628
Epoch  270  loss  0.5519482425863648 correct 50
Average time per epoch 1.1997459468841554 (for 500 epochs)
Epoch  271  time  1.9492475986480713
Average time per epoch 1.2036444420814514 (for 500 epochs)
Epoch  272  time  1.992387294769287
Average time per epoch 1.2076292166709899 (for 500 epochs)
Epoch  273  time  1.9818799495697021
Average time per epoch 1.2115929765701294 (for 500 epochs)
Epoch  274  time  1.9283461570739746
Average time per epoch 1.2154496688842773 (for 500 epochs)
Epoch  275  time  1.9696424007415771
Average time per epoch 1.2193889536857605 (for 500 epochs)
Epoch  276  time  2.035614490509033
Average time per epoch 1.2234601826667786 (for 500 epochs)
Epoch  277  time  1.9420006275177002
Average time per epoch 1.227344183921814 (for 500 epochs)
Epoch  278  time  1.9525861740112305
Average time per epoch 1.2312493562698363 (for 500 epochs)
Epoch  279  time  2.008164167404175
Average time per epoch 1.2352656846046448 (for 500 epochs)
Epoch  280  time  2.297781467437744
Epoch  280  loss  0.2513984640257759 correct 49
Average time per epoch 1.2398612475395203 (for 500 epochs)
Epoch  281  time  3.165729522705078
Average time per epoch 1.2461927065849303 (for 500 epochs)
Epoch  282  time  3.3990681171417236
Average time per epoch 1.2529908428192138 (for 500 epochs)
Epoch  283  time  3.451206922531128
Average time per epoch 1.2598932566642762 (for 500 epochs)
Epoch  284  time  2.505394220352173
Average time per epoch 1.2649040451049804 (for 500 epochs)
Epoch  285  time  1.9451277256011963
Average time per epoch 1.2687943005561828 (for 500 epochs)
Epoch  286  time  2.0275235176086426
Average time per epoch 1.2728493475914002 (for 500 epochs)
Epoch  287  time  1.9291985034942627
Average time per epoch 1.2767077445983888 (for 500 epochs)
Epoch  288  time  1.9470157623291016
Average time per epoch 1.2806017761230468 (for 500 epochs)
Epoch  289  time  1.947779655456543
Average time per epoch 1.28449733543396 (for 500 epochs)
Epoch  290  time  2.0084140300750732
Epoch  290  loss  1.1657433133107724 correct 48
Average time per epoch 1.2885141634941102 (for 500 epochs)
Epoch  291  time  1.9477272033691406
Average time per epoch 1.2924096179008484 (for 500 epochs)
Epoch  292  time  1.9526035785675049
Average time per epoch 1.2963148250579835 (for 500 epochs)
Epoch  293  time  2.004149913787842
Average time per epoch 1.3003231248855591 (for 500 epochs)
Epoch  294  time  1.9785387516021729
Average time per epoch 1.3042802023887634 (for 500 epochs)
Epoch  295  time  1.9400804042816162
Average time per epoch 1.3081603631973266 (for 500 epochs)
Epoch  296  time  1.9804551601409912
Average time per epoch 1.3121212735176087 (for 500 epochs)
Epoch  297  time  2.030245780944824
Average time per epoch 1.3161817650794982 (for 500 epochs)
Epoch  298  time  1.9221711158752441
Average time per epoch 1.3200261073112487 (for 500 epochs)
Epoch  299  time  1.9457848072052002
Average time per epoch 1.323917676925659 (for 500 epochs)
Epoch  300  time  1.9986090660095215
Epoch  300  loss  0.05327880178030844 correct 50
Average time per epoch 1.3279148950576782 (for 500 epochs)
Epoch  301  time  1.971785306930542
Average time per epoch 1.3318584656715393 (for 500 epochs)
Epoch  302  time  1.958914041519165
Average time per epoch 1.3357762937545776 (for 500 epochs)
Epoch  303  time  1.946854591369629
Average time per epoch 1.339670002937317 (for 500 epochs)
Epoch  304  time  2.0069572925567627
Average time per epoch 1.3436839175224304 (for 500 epochs)
Epoch  305  time  1.9511628150939941
Average time per epoch 1.3475862431526184 (for 500 epochs)
Epoch  306  time  1.9373559951782227
Average time per epoch 1.3514609551429748 (for 500 epochs)
Epoch  307  time  1.9998295307159424
Average time per epoch 1.3554606142044068 (for 500 epochs)
Epoch  308  time  1.9889326095581055
Average time per epoch 1.359438479423523 (for 500 epochs)
Epoch  309  time  1.9625506401062012
Average time per epoch 1.3633635807037354 (for 500 epochs)
Epoch  310  time  1.970160722732544
Epoch  310  loss  0.05120901853148611 correct 50
Average time per epoch 1.3673039021492004 (for 500 epochs)
Epoch  311  time  2.0299792289733887
Average time per epoch 1.3713638606071472 (for 500 epochs)
Epoch  312  time  1.987196683883667
Average time per epoch 1.3753382539749146 (for 500 epochs)
Epoch  313  time  1.9161972999572754
Average time per epoch 1.3791706485748292 (for 500 epochs)
Epoch  314  time  2.0191915035247803
Average time per epoch 1.3832090315818786 (for 500 epochs)
Epoch  315  time  2.3011226654052734
Average time per epoch 1.3878112769126891 (for 500 epochs)
Epoch  316  time  3.2235782146453857
Average time per epoch 1.39425843334198 (for 500 epochs)
Epoch  317  time  3.4243171215057373
Average time per epoch 1.4011070675849915 (for 500 epochs)
Epoch  318  time  3.4508864879608154
Average time per epoch 1.408008840560913 (for 500 epochs)
Epoch  319  time  2.4796395301818848
Average time per epoch 1.4129681196212769 (for 500 epochs)
Epoch  320  time  1.9326293468475342
Epoch  320  loss  0.31308790619676935 correct 48
Average time per epoch 1.416833378314972 (for 500 epochs)
Epoch  321  time  1.983332633972168
Average time per epoch 1.4208000435829162 (for 500 epochs)
Epoch  322  time  1.977574348449707
Average time per epoch 1.4247551922798156 (for 500 epochs)
Epoch  323  time  1.9543609619140625
Average time per epoch 1.4286639142036437 (for 500 epochs)
Epoch  324  time  1.975005865097046
Average time per epoch 1.432613925933838 (for 500 epochs)
Epoch  325  time  2.0215704441070557
Average time per epoch 1.436657066822052 (for 500 epochs)
Epoch  326  time  1.9579310417175293
Average time per epoch 1.4405729289054872 (for 500 epochs)
Epoch  327  time  1.9692411422729492
Average time per epoch 1.444511411190033 (for 500 epochs)
Epoch  328  time  2.0091819763183594
Average time per epoch 1.4485297751426698 (for 500 epochs)
Epoch  329  time  1.9601492881774902
Average time per epoch 1.4524500737190247 (for 500 epochs)
Epoch  330  time  1.9544661045074463
Epoch  330  loss  0.9264010035541542 correct 48
Average time per epoch 1.4563590059280396 (for 500 epochs)
Epoch  331  time  1.9568262100219727
Average time per epoch 1.4602726583480834 (for 500 epochs)
Epoch  332  time  2.0308279991149902
Average time per epoch 1.4643343143463134 (for 500 epochs)
Epoch  333  time  1.9571683406829834
Average time per epoch 1.4682486510276795 (for 500 epochs)
Epoch  334  time  1.9477691650390625
Average time per epoch 1.4721441893577576 (for 500 epochs)
Epoch  335  time  2.0157217979431152
Average time per epoch 1.4761756329536437 (for 500 epochs)
Epoch  336  time  1.9412579536437988
Average time per epoch 1.4800581488609315 (for 500 epochs)
Epoch  337  time  1.9360942840576172
Average time per epoch 1.4839303374290467 (for 500 epochs)
Epoch  338  time  1.922767162322998
Average time per epoch 1.4877758717536926 (for 500 epochs)
Epoch  339  time  2.0367188453674316
Average time per epoch 1.4918493094444274 (for 500 epochs)
Epoch  340  time  1.996568202972412
Epoch  340  loss  0.24882425067917358 correct 49
Average time per epoch 1.4958424458503723 (for 500 epochs)
Epoch  341  time  1.9470932483673096
Average time per epoch 1.4997366323471069 (for 500 epochs)
Epoch  342  time  1.995924711227417
Average time per epoch 1.5037284817695618 (for 500 epochs)
Epoch  343  time  1.930588960647583
Average time per epoch 1.507589659690857 (for 500 epochs)
Epoch  344  time  1.951322078704834
Average time per epoch 1.5114923038482666 (for 500 epochs)
Epoch  345  time  1.9505248069763184
Average time per epoch 1.5153933534622193 (for 500 epochs)
Epoch  346  time  2.0052692890167236
Average time per epoch 1.5194038920402526 (for 500 epochs)
Epoch  347  time  2.0382139682769775
Average time per epoch 1.5234803199768066 (for 500 epochs)
Epoch  348  time  1.9176580905914307
Average time per epoch 1.5273156361579896 (for 500 epochs)
Epoch  349  time  2.0262157917022705
Average time per epoch 1.5313680677413941 (for 500 epochs)
Epoch  350  time  2.857729196548462
Epoch  350  loss  0.0017241614719467598 correct 50
Average time per epoch 1.537083526134491 (for 500 epochs)
Epoch  351  time  3.438445806503296
Average time per epoch 1.5439604177474975 (for 500 epochs)
Epoch  352  time  3.38069486618042
Average time per epoch 1.5507218074798583 (for 500 epochs)
Epoch  353  time  3.2082014083862305
Average time per epoch 1.557138210296631 (for 500 epochs)
Epoch  354  time  1.9534375667572021
Average time per epoch 1.5610450854301452 (for 500 epochs)
Epoch  355  time  1.9542996883392334
Average time per epoch 1.5649536848068237 (for 500 epochs)
Epoch  356  time  2.0184988975524902
Average time per epoch 1.5689906826019286 (for 500 epochs)
Epoch  357  time  1.92338228225708
Average time per epoch 1.5728374471664428 (for 500 epochs)
Epoch  358  time  1.940131425857544
Average time per epoch 1.576717710018158 (for 500 epochs)
Epoch  359  time  1.964813470840454
Average time per epoch 1.5806473369598388 (for 500 epochs)
Epoch  360  time  2.005932569503784
Epoch  360  loss  0.5144578199007183 correct 49
Average time per epoch 1.5846592020988464 (for 500 epochs)
Epoch  361  time  1.9398677349090576
Average time per epoch 1.5885389375686645 (for 500 epochs)
Epoch  362  time  1.956437110900879
Average time per epoch 1.5924518117904662 (for 500 epochs)
Epoch  363  time  1.93693208694458
Average time per epoch 1.5963256759643554 (for 500 epochs)
Epoch  364  time  1.9859700202941895
Average time per epoch 1.6002976160049438 (for 500 epochs)
Epoch  365  time  1.916358232498169
Average time per epoch 1.60413033246994 (for 500 epochs)
Epoch  366  time  1.9263525009155273
Average time per epoch 1.6079830374717712 (for 500 epochs)
Epoch  367  time  1.9870727062225342
Average time per epoch 1.6119571828842163 (for 500 epochs)
Epoch  368  time  2.0033328533172607
Average time per epoch 1.6159638485908507 (for 500 epochs)
Epoch  369  time  1.935734510421753
Average time per epoch 1.6198353176116944 (for 500 epochs)
Epoch  370  time  1.9843571186065674
Epoch  370  loss  0.42736211654605 correct 50
Average time per epoch 1.6238040318489075 (for 500 epochs)
Epoch  371  time  1.9227797985076904
Average time per epoch 1.6276495914459228 (for 500 epochs)
Epoch  372  time  1.9256963729858398
Average time per epoch 1.6315009841918946 (for 500 epochs)
Epoch  373  time  1.9317796230316162
Average time per epoch 1.6353645434379578 (for 500 epochs)
Epoch  374  time  2.1417031288146973
Average time per epoch 1.6396479496955871 (for 500 epochs)
Epoch  375  time  3.094785690307617
Average time per epoch 1.6458375210762024 (for 500 epochs)
Epoch  376  time  3.3105368614196777
Average time per epoch 1.6524585947990418 (for 500 epochs)
Epoch  377  time  3.418325185775757
Average time per epoch 1.6592952451705933 (for 500 epochs)
Epoch  378  time  2.8412392139434814
Average time per epoch 1.6649777235984802 (for 500 epochs)
Epoch  379  time  1.9464070796966553
Average time per epoch 1.6688705377578736 (for 500 epochs)
Epoch  380  time  1.9607598781585693
Epoch  380  loss  0.6358912540677893 correct 50
Average time per epoch 1.6727920575141906 (for 500 epochs)
Epoch  381  time  2.1364967823028564
Average time per epoch 1.6770650510787963 (for 500 epochs)
Epoch  382  time  3.021111011505127
Average time per epoch 1.6831072731018066 (for 500 epochs)
Epoch  383  time  3.4380998611450195
Average time per epoch 1.6899834728240968 (for 500 epochs)
Epoch  384  time  3.465322256088257
Average time per epoch 1.6969141173362732 (for 500 epochs)
Epoch  385  time  3.1783175468444824
Average time per epoch 1.7032707524299622 (for 500 epochs)
Epoch  386  time  1.935105800628662
Average time per epoch 1.7071409640312194 (for 500 epochs)
Epoch  387  time  1.9519264698028564
Average time per epoch 1.7110448169708252 (for 500 epochs)
Epoch  388  time  2.0295214653015137
Average time per epoch 1.7151038599014283 (for 500 epochs)
Epoch  389  time  1.944958209991455
Average time per epoch 1.7189937763214111 (for 500 epochs)
Epoch  390  time  1.9564228057861328
Epoch  390  loss  0.3931506759398001 correct 50
Average time per epoch 1.7229066219329834 (for 500 epochs)
Epoch  391  time  1.9467930793762207
Average time per epoch 1.7268002080917357 (for 500 epochs)
Epoch  392  time  2.023122549057007
Average time per epoch 1.7308464531898498 (for 500 epochs)
Epoch  393  time  1.9372432231903076
Average time per epoch 1.7347209396362304 (for 500 epochs)
Epoch  394  time  2.024698257446289
Average time per epoch 1.738770336151123 (for 500 epochs)
Epoch  395  time  1.999809980392456
Average time per epoch 1.7427699561119079 (for 500 epochs)
Epoch  396  time  1.916348934173584
Average time per epoch 1.746602653980255 (for 500 epochs)
Epoch  397  time  1.960343837738037
Average time per epoch 1.750523341655731 (for 500 epochs)
Epoch  398  time  2.31459903717041
Average time per epoch 1.755152539730072 (for 500 epochs)
Epoch  399  time  3.222087860107422
Average time per epoch 1.761596715450287 (for 500 epochs)
Epoch  400  time  3.337229013442993
Epoch  400  loss  0.3238415862655154 correct 50
Average time per epoch 1.7682711734771728 (for 500 epochs)
Epoch  401  time  3.3175151348114014
Average time per epoch 1.7749062037467958 (for 500 epochs)
Epoch  402  time  2.5431156158447266
Average time per epoch 1.779992434978485 (for 500 epochs)
Epoch  403  time  1.96415376663208
Average time per epoch 1.7839207425117494 (for 500 epochs)
Epoch  404  time  1.9762279987335205
Average time per epoch 1.7878731985092162 (for 500 epochs)
Epoch  405  time  1.9620575904846191
Average time per epoch 1.7917973136901855 (for 500 epochs)
Epoch  406  time  2.0142698287963867
Average time per epoch 1.7958258533477782 (for 500 epochs)
Epoch  407  time  2.05598521232605
Average time per epoch 1.7999378237724304 (for 500 epochs)
Epoch  408  time  1.9412071704864502
Average time per epoch 1.8038202381134034 (for 500 epochs)
Epoch  409  time  2.005034923553467
Average time per epoch 1.8078303079605103 (for 500 epochs)
Epoch  410  time  1.9583640098571777
Epoch  410  loss  0.02092828597291675 correct 45
Average time per epoch 1.8117470359802246 (for 500 epochs)
Epoch  411  time  1.9408953189849854
Average time per epoch 1.8156288266181946 (for 500 epochs)
Epoch  412  time  1.956449270248413
Average time per epoch 1.8195417251586914 (for 500 epochs)
Epoch  413  time  3.005316734313965
Average time per epoch 1.8255523586273192 (for 500 epochs)
Epoch  414  time  3.315342426300049
Average time per epoch 1.8321830434799193 (for 500 epochs)
Epoch  415  time  3.336644172668457
Average time per epoch 1.8388563318252564 (for 500 epochs)
Epoch  416  time  3.196289539337158
Average time per epoch 1.8452489109039307 (for 500 epochs)
Epoch  417  time  1.945953369140625
Average time per epoch 1.849140817642212 (for 500 epochs)
Epoch  418  time  1.9292340278625488
Average time per epoch 1.852999285697937 (for 500 epochs)
Epoch  419  time  1.9447426795959473
Average time per epoch 1.856888771057129 (for 500 epochs)
Epoch  420  time  2.125448703765869
Epoch  420  loss  0.000103379826323845 correct 50
Average time per epoch 1.8611396684646606 (for 500 epochs)
Epoch  421  time  1.9253404140472412
Average time per epoch 1.864990349292755 (for 500 epochs)
Epoch  422  time  1.9442646503448486
Average time per epoch 1.868878878593445 (for 500 epochs)
Epoch  423  time  2.0306506156921387
Average time per epoch 1.872940179824829 (for 500 epochs)
Epoch  424  time  1.955169916152954
Average time per epoch 1.876850519657135 (for 500 epochs)
Epoch  425  time  1.9283254146575928
Average time per epoch 1.8807071704864502 (for 500 epochs)
Epoch  426  time  1.9838438034057617
Average time per epoch 1.8846748580932617 (for 500 epochs)
Epoch  427  time  2.0312275886535645
Average time per epoch 1.888737313270569 (for 500 epochs)
Epoch  428  time  1.9672391414642334
Average time per epoch 1.8926717915534974 (for 500 epochs)
Epoch  429  time  1.93552565574646
Average time per epoch 1.8965428428649902 (for 500 epochs)
Epoch  430  time  2.0007381439208984
Epoch  430  loss  0.03888298912701214 correct 50
Average time per epoch 1.900544319152832 (for 500 epochs)
Epoch  431  time  1.9626600742340088
Average time per epoch 1.9044696393013 (for 500 epochs)
Epoch  432  time  1.9517250061035156
Average time per epoch 1.9083730893135071 (for 500 epochs)
Epoch  433  time  1.9573142528533936
Average time per epoch 1.912287717819214 (for 500 epochs)
Epoch  434  time  2.016507625579834
Average time per epoch 1.9163207330703735 (for 500 epochs)
Epoch  435  time  1.9965629577636719
Average time per epoch 1.9203138589859008 (for 500 epochs)
Epoch  436  time  1.9343926906585693
Average time per epoch 1.924182644367218 (for 500 epochs)
Epoch  437  time  2.041522264480591
Average time per epoch 1.928265688896179 (for 500 epochs)
Epoch  438  time  1.924241065979004
Average time per epoch 1.9321141710281373 (for 500 epochs)
Epoch  439  time  1.9480125904083252
Average time per epoch 1.9360101962089538 (for 500 epochs)
Epoch  440  time  1.9472146034240723
Epoch  440  loss  0.032656441539494825 correct 50
Average time per epoch 1.939904625415802 (for 500 epochs)
Epoch  441  time  1.9985482692718506
Average time per epoch 1.9439017219543457 (for 500 epochs)
Epoch  442  time  1.9089899063110352
Average time per epoch 1.9477197017669678 (for 500 epochs)
Epoch  443  time  1.9105072021484375
Average time per epoch 1.9515407161712646 (for 500 epochs)
Epoch  444  time  2.0319111347198486
Average time per epoch 1.9556045384407044 (for 500 epochs)
Epoch  445  time  1.9120886325836182
Average time per epoch 1.9594287157058716 (for 500 epochs)
Epoch  446  time  1.9264576435089111
Average time per epoch 1.9632816309928893 (for 500 epochs)
Epoch  447  time  1.9523425102233887
Average time per epoch 1.9671863160133363 (for 500 epochs)
Epoch  448  time  3.1986024379730225
Average time per epoch 1.9735835208892822 (for 500 epochs)
Epoch  449  time  3.472191333770752
Average time per epoch 1.9805279035568237 (for 500 epochs)
Epoch  450  time  3.381699562072754
Epoch  450  loss  0.12504917185536524 correct 50
Average time per epoch 1.9872913026809693 (for 500 epochs)
Epoch  451  time  2.884570837020874
Average time per epoch 1.993060444355011 (for 500 epochs)
Epoch  452  time  1.953169584274292
Average time per epoch 1.9969667835235596 (for 500 epochs)
Epoch  453  time  1.957162857055664
Average time per epoch 2.000881109237671 (for 500 epochs)
Epoch  454  time  1.9758999347686768
Average time per epoch 2.004832909107208 (for 500 epochs)
Epoch  455  time  2.010399580001831
Average time per epoch 2.0088537082672118 (for 500 epochs)
Epoch  456  time  1.9319970607757568
Average time per epoch 2.0127177023887635 (for 500 epochs)
Epoch  457  time  1.951716661453247
Average time per epoch 2.01662113571167 (for 500 epochs)
Epoch  458  time  1.9840548038482666
Average time per epoch 2.0205892453193663 (for 500 epochs)
Epoch  459  time  1.9798741340637207
Average time per epoch 2.024548993587494 (for 500 epochs)
Epoch  460  time  1.9863762855529785
Epoch  460  loss  0.15789807548593437 correct 50
Average time per epoch 2.0285217461586 (for 500 epochs)
Epoch  461  time  1.9429550170898438
Average time per epoch 2.0324076561927797 (for 500 epochs)
Epoch  462  time  2.029395580291748
Average time per epoch 2.036466447353363 (for 500 epochs)
Epoch  463  time  1.9418439865112305
Average time per epoch 2.0403501353263853 (for 500 epochs)
Epoch  464  time  1.9783680438995361
Average time per epoch 2.0443068714141845 (for 500 epochs)
Epoch  465  time  2.0455105304718018
Average time per epoch 2.048397892475128 (for 500 epochs)
Epoch  466  time  1.9327518939971924
Average time per epoch 2.0522633962631227 (for 500 epochs)
Epoch  467  time  1.9350550174713135
Average time per epoch 2.0561335062980652 (for 500 epochs)
Epoch  468  time  1.9640448093414307
Average time per epoch 2.060061595916748 (for 500 epochs)
Epoch  469  time  1.9912183284759521
Average time per epoch 2.0640440325737 (for 500 epochs)
Epoch  470  time  1.915482759475708
Epoch  470  loss  0.26409478356363286 correct 50
Average time per epoch 2.0678749980926514 (for 500 epochs)
Epoch  471  time  1.937434434890747
Average time per epoch 2.071749866962433 (for 500 epochs)
Epoch  472  time  1.9916610717773438
Average time per epoch 2.0757331891059874 (for 500 epochs)
Epoch  473  time  1.9496831893920898
Average time per epoch 2.0796325554847717 (for 500 epochs)
Epoch  474  time  1.9372034072875977
Average time per epoch 2.0835069622993467 (for 500 epochs)
Epoch  475  time  1.9464852809906006
Average time per epoch 2.087399932861328 (for 500 epochs)
Epoch  476  time  2.0130770206451416
Average time per epoch 2.091426086902618 (for 500 epochs)
Epoch  477  time  1.9323792457580566
Average time per epoch 2.0952908453941346 (for 500 epochs)
Epoch  478  time  1.9364447593688965
Average time per epoch 2.0991637349128722 (for 500 epochs)
Epoch  479  time  2.0124404430389404
Average time per epoch 2.1031886157989503 (for 500 epochs)
Epoch  480  time  2.0397653579711914
Epoch  480  loss  0.1767188282461838 correct 50
Average time per epoch 2.1072681465148926 (for 500 epochs)
Epoch  481  time  2.6347155570983887
Average time per epoch 2.1125375776290896 (for 500 epochs)
Epoch  482  time  3.337977170944214
Average time per epoch 2.1192135319709777 (for 500 epochs)
Epoch  483  time  3.4383716583251953
Average time per epoch 2.1260902752876283 (for 500 epochs)
Epoch  484  time  3.383443593978882
Average time per epoch 2.132857162475586 (for 500 epochs)
Epoch  485  time  2.0007505416870117
Average time per epoch 2.13685866355896 (for 500 epochs)
Epoch  486  time  2.0039827823638916
Average time per epoch 2.1408666291236877 (for 500 epochs)
Epoch  487  time  2.014106512069702
Average time per epoch 2.1448948421478273 (for 500 epochs)
Epoch  488  time  2.010864019393921
Average time per epoch 2.148916570186615 (for 500 epochs)
Epoch  489  time  1.9739937782287598
Average time per epoch 2.1528645577430727 (for 500 epochs)
Epoch  490  time  2.0220649242401123
Epoch  490  loss  0.2109435280856573 correct 50
Average time per epoch 2.1569086875915526 (for 500 epochs)
Epoch  491  time  1.9538662433624268
Average time per epoch 2.1608164200782776 (for 500 epochs)
Epoch  492  time  1.946603775024414
Average time per epoch 2.1647096276283264 (for 500 epochs)
Epoch  493  time  2.076538562774658
Average time per epoch 2.1688627047538755 (for 500 epochs)
Epoch  494  time  1.9341614246368408
Average time per epoch 2.1727310276031493 (for 500 epochs)
Epoch  495  time  1.9489624500274658
Average time per epoch 2.1766289525032043 (for 500 epochs)
Epoch  496  time  1.9710378646850586
Average time per epoch 2.1805710282325745 (for 500 epochs)
Epoch  497  time  2.0420172214508057
Average time per epoch 2.184655062675476 (for 500 epochs)
Epoch  498  time  1.963057518005371
Average time per epoch 2.188581177711487 (for 500 epochs)
Epoch  499  time  1.9738194942474365
Average time per epoch 2.1925288166999817 (for 500 epochs)
```


# Plot:

* Note:

Several of the tests for this assignment will only run if you are on a GPU machine and will not
run on github's test infrastructure. Please follow the instructions to setup up a colab machine
to run these tests.

This assignment requires the following files from the previous assignments. You can get these by running

```bash
python sync_previous_module.py previous-module-dir current-module-dir
```

The files that will be synced are:

        minitorch/tensor_data.py minitorch/tensor_functions.py minitorch/tensor_ops.py minitorch/operators.py minitorch/module.py minitorch/autodiff.py minitorch/module.py project/run_manual.py project/run_scalar.py project/run_tensor.py
