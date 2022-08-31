from flask import Flask, render_template, request, flash
from recommender import recommend_random, nmf_predict, prepare_new_user
from utils import movies, fuzzy_title_to_id, id_to_title, fuzzy_title_to_id_only, fuzzy_title_to_clean_title


app = Flask(__name__) # Initialise app 

app.secret_key = b"need_this_for_message_flashing"

# Landing page 
@app.route("/")
@app.route("/landing_page")
def landing_page():
    return render_template("landing_page.html")

@app.route('/about/')
def about():
    return render_template('about.html')

@app.route('/models/')
def models():
    return render_template('models.html')

@app.route('/contact/')
def contact():
    return render_template('contact.html')

# Recommendations page
@app.route('/recommender/')
def recommender():

    user_inputs = request.args.getlist('user_movies')

    titles_clean = fuzzy_title_to_clean_title(movies_df=movies, title_from_user = user_inputs)
    # titles_ids= [fuzzy_title_to_id(movies_df = movies, title_from_user = user_input) for user_input in user_inputs]
    ids_only = [fuzzy_title_to_id_only(movies_df = movies, title_from_user = user_input) for user_input in user_inputs]
    
    '''
    chosen_model = request.args.get('model')

    if chosen_model == 'nmf':
        recs = nmf_predict(new_user=prepare_new_user(user_inputs))
    elif chosen_model == 'random':
        recs = recommend_random()
    '''

    recs = nmf_predict(new_user=prepare_new_user(user_inputs))
    #[id_to_title(movies_df = movies, movie_id=id) for id in rec_ids]

    #print(recs)

    #if request.args.getlist('user_model') == 'NMF':
     #   nmf_model(ids_only)

    return render_template('recommender.html', user_inputs=user_inputs, ids_only=ids_only, titles_ids = titles_clean, recs=recs)



if __name__ == "__main__":
    app.run(debug=True)
