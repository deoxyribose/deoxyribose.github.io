---
layout: post
title: "On form and formlessness"
subtitle: "A metaphor for non-duality, borrowed from computer science"
---

*A speech for a philosophy meetup, translated from Danish by Claude. The Danish original is below.*

Everything we ever experience — sense impressions, thoughts, feelings, concepts — arises in our consciousness in a given form, imbued with a particular meaning. For example, an apple has a round, red, and sweet form. An apple can mean nourishment if we are hungry, a product if we are out shopping, or a weapon if we are angry.

Formlessness is the absence of form, and the absence of meaning. When we don't know how to make sense of a situation, we are met with formlessness. When our concepts fail to describe the reality we live in, that is formlessness. Formlessness is also potential — it is out of the formless that all forms arise. Before a child learns what an apple is, it is a collection of sense impressions — red, sweet, juicy. Before a university student can grasp a new and perhaps difficult concept, it is intangible, confusing — meaningless.

It is a doctrine in Eastern philosophy that form and formlessness are not really opposites — one speaks of the non-duality of form and formlessness. In some way, the doctrine goes, form and formlessness are inseparable, one and the same thing. It is so abstract and mystical that one is usually advised to give up on understanding it intellectually, and instead simply to train in a given tradition until one directly experiences it.

In this talk I will try to offer a metaphor for this non-duality — a metaphor I borrow from computer science.

Slightly reformulated, it is a doctrine in computer science that there is a non-duality between data and programs.

That probably wasn't much more comprehensible, but we'll take it one step at a time. Let me explain what data is, and then what programs are. I'll give two programs as an example, which take the same data as input but produce different output.

```
add_two_numbers(X, Y):
    result = X
    for 1..Y
        result = result + 1

add_two_numbers(5, 3)
```

```
make_scrambled_eggs(num_guests, eggs_per_guest):
    num_eggs = eggs_per_guest * num_guests
    amount_of_salt = eggs_per_guest * 1/4 teaspoon salt
    for egg = 1..num_eggs
        throw 1 egg on the pan
    throw amount_of_salt salt on the pan
    as long as there are still raw eggs:
        stir

make_scrambled_eggs(5, 3)
```

In both cases we have the same data: 5 and 3. In themselves those numbers mean nothing — they are meaningless. It is only when you run a program, with that data as input, that they are interpreted and acquire meaning. And they acquire completely different meanings depending on which program I run — in fact there are infinitely many ways to interpret them, because you can write infinitely many different programs.

But I began by claiming that there is no difference between data and programs. What do I mean by that?

When I read my two programs aloud, you understand what I mean, because we all speak Danish, and you know what it means to add 1 to a number, you know what an egg is, and what it means to stir something.

But I can't do the same with a computer, because it doesn't speak Danish, and doesn't understand what an egg is. To run the programs, I first have to write an interpreter or a compiler. An interpreter is a program that takes a program you've written as input, and outputs instructions that your computer can carry out.

For example, an interpreter might contain things like:

When the program says "X * Y", add X to itself Y times.
When you see "throw X objects on the pan", figure out where the object is, take X of them in your hand, figure out where the pan is, move your hand over the pan, and let go of the objects.

But you could just as well write another interpreter, which said things like:

When you see "X * Y", add Y to itself X times (in this case the result is the same, but the method is different).
When you see "throw X objects on the pan", figure out where your eyebrows are, go up a couple of centimeters, lie down, and place one object at a time on your forehead until there are X of them. That is at any rate an interpretation that leads to a different result.

So in the same way as it was with data, a program is meaningless until it is interpreted. And there are infinitely many ways to interpret a program. By the way, if you've ever had to choose between Windows 64-bit and 32-bit, that's what you were choosing — different ways of interpreting programs so that your computer can carry them out. It's because programs simply *are* data. And you can just as well say that all data is programs — they just have to be interpreted correctly.

Now comes my philosophical claim: the relation we have between data and programs is precisely the same relation that holds for formlessness and form. Formlessness is like data — without meaning, intangible, but at the same time the potential that forms, or programs, are made of. Form is like programs — they are given meaning by the right interpretation, but they can be interpreted in many ways, and thereby used for different things.

Fine. Let's say we believe it. How can one use this insight in practice? Here again one can draw a parallel to computer science. It is useful for a programmer to see a program as data, to see data as a program, and to switch flexibly between the two perspectives. It can lead to elegant ways of solving problems; you can even get a program to edit its own source code, and change what it does while it's running.

In our daily lives, it can be very useful to be able to take extremely meaningful experiences — a goal we've worked toward for ten years, the end of a relationship, or perhaps intense physical pain — and see them as formless. If you can realize that those experiences are, in a sense, just data — it opens up other ways of relating to them.

And it can also be useful to be able to take meaningless experiences — a bus ride, standing in line at the supermarket, being dead tired — and look at their form. How do these things feel? The bus ride is meaningless routine, but isn't it actually completely wild to be able to move faster than any animal without exerting yourself? It's not great to be dead tired — but isn't there something pleasant about it too, somewhere? Maybe. It's up to interpretation.

Cheers, in spirit.

---

## Dansk original

Alt hvad vi nogensinde oplever - sanseindtryk, tanker, følelser, koncepter - opstår i vores bevidsthed i en given form, tillagt en bestemt mening. For eksempel, et æble har en rund, rød og sød form. Et æble kan betyde næring hvis vi er sultne, et produkt hvis vi er ude og handle, eller et våben hvis vi er vrede.

Formløshed er mangel på form, og mangel på mening. Når vi ikke ved hvordan vi skal forstå en situation, bliver vi mødt med formløshed. Når vores koncepter ikke formår at beskrive den virkelighed vi lever i, er det formløshed. Formløshed er også potentiale - det er ud af det formløse alle former opstår. Før er barn lærer hvad et æble er, er det en samling af sanseindtryk - rød, sød, saftig. Før en universitetsstudent kan begribe et nyt og måske svært koncept, er det uhåndgribeligt, forvirrende - meningsløst.

Det er en doktrine i Østlig filosofi at form og formløshed i virkeligheden ikke er modsætninger - man taler om ikke-dualiteten af form og formløshed. På en eller anden, lyder doktrinen, er form og formløshed uadskillelige, en og samme ting. Det er så abstrakt og mystisk, at man normalt anbefaler at give op på at forstå det intellektuelt, og i stedet for bare træne i en given tradition indtil man direkte oplever det.

Jeg vil i denne tale forsøge at komme med en metafor for denne ikke-dualitet - en metafor jeg låner fra datalogi.

Lidt omformuleret, er det en doktrine i datalogi, at der er en ikke-dualitet mellem data og programmer.

Det var nok ikke meget mere forståeligt, men vi tager det et skridt af gangen. Lad mig forklare hvad data er, og så hvad programmer er. Jeg vil give to programmer som eksempel, som tager samme data som input, men giver forskelligt output.

```
læg_to_tal_sammen(X, Y):
    resultat = X
    for 1..Y
        resultat = resultat + 1

læg_to_tal_sammen(5, 3)
```

```
lav_røræg(antal gæster, antal_æg_per_gæst):
    antal_æg = antal_æg_per_gæst * antal_gæster
    mængde_salt = antal_æg_per_gæst * 1/4 teske salt
    for æg = 1..antal_æg
        smid 1 æg på panden
    smid mængde_salt salt på panden
    så længe der stadig er rå æg:
        rør rundt

lav_røræg(5, 3)
```

I begge tilfælde har vi samme data: 5 og 3. I sig selv betyder de tal ikke noget - de er meningsløse. Det er kun idet man kører et program, med det data som input, at de bliver fortolket, og får mening. Og de får helt forskellige meninger afhængigt af hvilket program jeg kører - der er faktisk uendeligt mange måder at fortolke dem på, fordi man kan skrive uendeligt mange forskellige programmer.

Men jeg startede med at påstå at der ikke er nogen forskel på data og programmer. Hvad mener jeg med det?

Når jeg læser mine to programmer op, forstår I hvad jeg mener, fordi vi alle taler dansk, og I ved hvad det betyder at lægge 1 til et tal, I ved hvad et æg er, og hvad det vil sige at røre rundt i noget.

Men jeg kan ikke gøre det samme med en computer, fordi den taler ikke dansk, og forstår ikke hvad et æg er. For at køre programmerne, er jeg først nødt til at skrive en intepreter eller compiler. En interpreter er et program, som tager et program du har skrevet som input, og outputter instrukser som din computer kan udføre.

For eksempel, en fortolker kunne indeholde ting som:
Når programmet siger "X * Y", læg X oven i sig selv Y gange.
Når du ser "smid X objekter på panden", find ud af hvor objekt er, tag X af dem i hånden, find ud af hvor panden er, før hånden hen over panden, og giv slip på objekterne.

Men man kunne sagtens skrive en anden fortolker, som sagde ting som
Når du ser "X * Y", læg Y oven i sig selv X gange (i det her tilfælde er resultet det samme, men metoden er anderledes).
Når du ser "smid X objekter på panden", find ud af hvor dine øjenbryn er, gå et par centimeter op, læg dig ned, og læg et objekt ad gangen oven på din pande, indtil der er X af dem. Det er i hvertfald en fortolkning der fører til et anderledes resultat.

Så på samme måde som det var med data, et program er meningsløst, indtil det bliver fortolket. Og der er uendeligt mange måde at fortolke et program på. I øvrigt, hvis I nogensinde skulle vælge mellem Windows 64-bit eller 32-bit, så er det det I har valgt - forskellige måder at fortolke programmer så, så jeres computer kan udføre dem. Det er fordi programmer bare *er* data. Og man kan lige så godt sige, alt data er programmer - de skal bare fortolkes rigtigt.

Nu kommer min filosofiske påstand: den relation vi har mellem data og programmer, er præcis den samme relation som gælder for formløshed og form. Formløshed er ligesom data - uden betydning, uhåndgribeligt, men samtidigt det potentiale som former eller programmer består af. Form er ligesom programmer - de bliver tillagt mening givet den rette fortolkning, men de fortolkes på mange måder, og dermed bruges til andre ting.

Fint nok. Lad os sige vi tror på det. Hvordan kan man bruge den her indsigt i praksis? Her kan man igen drage parallel til datalogi. Det er nyttigt for en programmør at se på et program som data, se på data som et program, og skifte fleksibelt mellem de to perspektiver. Det kan føre til elegante måder at løse problemer på; man kan endda få et program til at redigere sin egen kildekode, og ændre hvad det gør mens det kører.

I vores daglige liv, kan det være meget nyttigt at kunne tage ekstremt meningsfyldte oplevelser - et mål vi har arbejdet hen imod i ti år, slutningen på et forhold eller måske kraftig fysisk smerte - og se på dem som formløse. Hvis man kan indse at de oplevelser, i en vis forstand, bare er data - åbner det op for andre måder at forholde sig til dem på.

Og det kan også være nyttigt at kunne tage meningsløse oplevelser - en bustur, et stå i kø i supermarkedet, at være dødsenstræt - og se på deres form. Hvordan føles de her ting? Busturen er meningsløs rutine, men er det egentligt ikke totalt langt ude at kunne bevæge sig hurtigere end noget dyr uden at anstrenge sig? Det er ikke fedt at være dødsenstræt - men er der ikke noget behageligt ved det også, et eller andet sted? Måske. Det er op til fortolkning.

Skål i ånd.
