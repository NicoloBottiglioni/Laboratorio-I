#Con il canellwtto si possono inserire i commmenti, i quali sono utili per sapere cosa si sta facendo
"""Per i commenti lunghi più di una riga di codice
si usano i tre apici, dopppi o singoli """

'''ciao
come stai '''

#Le funzioni si definiscono   nomefunzione('hvbeuv'). Una stringa di testo va sempre inserita tra apici singoli o doppi apici. Un numero, ivece, nella funzione print, può anche essere messo senza apici. Vale la stessa cosa anche per  le altre funzion?


print('ciao come state?')
print(43578634)
print('ciaoooo', 5654)


""" Una variabile è un nome/un simbolo che si da a una stringa o a un numero. Non hanno bisongo di essere prima definite e per assegnare loro un valore è necessario usare il simbolo =
"""

a= 'ciao come stai'
print(a)

b= 6
print(b)
print(a, b)

numerointero=8
g=8.
#adesso per definire l'argomento della funzine print, che vale anche per altre funzioni, si puossono utilizzare  due diverse tipologie di scritture

print('numero intero', numerointero, 'lettera a caso:', g)
#dopo aver definito la variabile numerointero e la variabile g, ho utilizzato la funzione print. Nell'argomento della funzione ho inserito una stringa di codicee tra virgolette seguita dai due punti. Ciò siggnifica, come si  può vedere dall'output, che la funzione mi restituisce ciò che ho scritto nell'argomento. In generale, una stringa va semore messsa tra virgolette (in questo caso il testo che  io voglio stampare), mentre una variabile  che  ho definito ouò  anche non essere messa tra virgolette

#La stessa funzione posso definirla anche nel seguente modo
print(f'numero intero: numerointero lettera a caso g')

#La differenza è che con f dopo posso mettere tutto, sia stringhe sia variabili, in una racchiuse una sola in degli apici

#le variabili possono essere anche delle liste e molto altro. Con la funzione type()  posso sapere di che  tipo è la variabile. Le liste vanno messe tra parentesi quadre e, se uno dei loro elementi è una stringa, devo sempre metterlo tra virgolette

c= 6
l=7.
lista=[1, 2, 3, 'cane']
animali=['gatto', 'mucca']

print(3, type('cane'))
print( type(c))
print(c, l)
print(f'gatto, type(gatto)')


