/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#include "helper_functions.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    // Number of particles to draw
	num_particles = 101;

    default_random_engine gen;

    // Extract standard deviations for x, y, and theta (initialization noise)
    double std_x = std[0];
    double std_y = std[1];
    double std_theta = std[2];

    // Create normal distributions for x, y and theta
    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    for (int i = 0; i < num_particles; ++i) {
        Particle p;

        // Sample  and from these normal distrubtions like this:
        //	 sample_x = dist_x(gen);
        //	 where "gen" is the random engine initialized earlier.

        p.id = i;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_theta(gen);
        p.weight = 1.0;

        // init particles
        particles.push_back(p);
        weights.push_back(p.weight);
    }
    // Flag, if filter is initialized
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    default_random_engine gen;

    for (int i = 0; i < num_particles; i++) {
        double particle_current_x = particles[i].x;
        double particle_current_y = particles[i].y;
        double particle_current_theta = particles[i].theta;

        double particle_predict_x = 0.0;
        double particle_predict_y = 0.0;
        double particle_predict_theta = 0.0;

        //be careful of the difference when theta is or is not zero
        if (fabs(yaw_rate) < 0.0001) {
            particle_predict_x = particle_current_x + velocity * cos(particle_current_theta) * delta_t;
            particle_predict_y = particle_current_y + velocity * sin(particle_current_theta) * delta_t;
            particle_predict_theta = particle_current_theta;
        } else {
            particle_predict_x = particle_current_x +
                    (velocity / yaw_rate) * (sin(particle_current_theta + (yaw_rate * delta_t)) - sin(particle_current_theta));
            particle_predict_y = particle_current_y +
                    (velocity / yaw_rate) * (-cos(particle_current_theta + (yaw_rate * delta_t)) + cos(particle_current_theta));
            particle_predict_theta = particle_current_theta +
                    (yaw_rate * delta_t);
        }

        // Extract standard deviations for x, y, and theta (sensor noise)
        double std_x = std_pos[0];
        double std_y = std_pos[1];
        double std_theta = std_pos[2];

        normal_distribution<double> dist_x(particle_predict_x, std_x);
        normal_distribution<double> dist_y(particle_predict_y, std_y);
        normal_distribution<double> dist_theta(particle_predict_theta, std_theta);

        particles[i].x = dist_x(gen);
        particles[i].y = dist_y(gen);
        particles[i].theta = dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted,
        std::vector<LandmarkObs>& observations,
        double sensor_range) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    for (size_t i = 0; i < observations.size(); i++) {
        //Init min distance to sqrt(2) times sensor_range. For those points that are out of this point, we
        // don't need to consider
        double min_dist = sensor_range * sqrt(2);
        //Init the id of landmark associated to an observation
        int landmark_id = -1;
        //Get current observation
        LandmarkObs o = observations[i];

        for (size_t j = 0; j < predicted.size(); j++) {
            //Get current prediction
            LandmarkObs p = predicted[j];

            //Get distance between current observation and predicted landmarks
            double current_dist = dist(o.x, o.y, p.x, p.y);

            if (current_dist < min_dist) {
                min_dist = current_dist;
                landmark_id = p.id;
            }
        }
        observations[i].id = landmark_id;
    }

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    // for each particle...
    for (int i = 0; i < num_particles; i++) {

        //Get x, y, theta coordinates of a particle based on the vehicle coordinates
        double particle_x = particles[i].x;
        double particle_y = particles[i].y;
        double particle_theta = particles[i].theta;

        /*****************************************************************************
        * Step 1: Transform observations from vehicle coordinates to map coordinates
        ****************************************************************************/

        //Create a vector to keep transformed observations
        vector<LandmarkObs> observations_tranformed;

        for (size_t j = 0; j < observations.size(); j++) {
            LandmarkObs obs;

            obs.id = observations[j].id;
            obs.x = particle_x +
                    cos(particle_theta) * observations[j].x - sin(particle_theta) * observations[j].y;
            obs.y = particle_y +
                    sin(particle_theta) * observations[j].x + cos(particle_theta) * observations[j].y;
            observations_tranformed.push_back(obs);
        }

        /*****************************************************************************
        * Step 2: Associate data with nearest neighbourhood method
        ****************************************************************************/

        /**
         2.1 Keep data within sensor_range
        **/
        //Create a vector to keep predictions within sensor range
        vector<LandmarkObs> landmarks_predicted;

        for (size_t j = 0; j < map_landmarks.landmark_list.size(); j++) {
            LandmarkObs lm;
            Map::single_landmark_s current_landmark = map_landmarks.landmark_list[j];

            lm.x = map_landmarks.landmark_list[j].x_f;
            lm.y = map_landmarks.landmark_list[j].y_f;
            lm.id = map_landmarks.landmark_list[j].id_i;

            //we use a rectangular region rather then circular region considering the computation speed
            if (fabs(lm.x - particle_x) <= sensor_range && fabs(lm.y - particle_y) <= sensor_range) {
                // add prediction to vector
                landmarks_predicted.push_back(lm);
            }
        }

        /**
          2.2 Perform data association using nearest neighborhood method
         **/
         //After this step, id in vector of observations_tranformed will be updated
        dataAssociation(landmarks_predicted, observations_tranformed);

        /*****************************************************************************
        * Step 3: Update weight wieh multivariate-Gaussian probability
        ****************************************************************************/

        //Define inputs to Gaussian probability
        double sigma_x = std_landmark[0];
        double sigma_y = std_landmark[1];
        //Calculate normalization term
        double gauss_norm = (1/(2 * M_PI * sigma_x * sigma_y));




        //Re-init weight of particles
        particles[i].weight = 1.0;

        for (size_t j = 0; j < observations_tranformed.size(); j++) {

            //Get observation coordinates and id
            double obs_x = observations_tranformed[j].x;
            double obs_y = observations_tranformed[j].y;
            int obs_id = observations_tranformed[j].id;

            //Get the x,y coordinates of the prediction associated with the current observation
            for (size_t k = 0; k < landmarks_predicted.size(); k++) {
                //Get prediction coordinates and id
                double pred_x = landmarks_predicted[k].x;
                double pred_y = landmarks_predicted[k].y;
                int pred_id = landmarks_predicted[k].id;

                if (pred_id == obs_id) {
                    //Calculate exponent
                    double exponent = (pow(obs_x - pred_x, 2))/(2 * pow(sigma_x, 2))
                            + (pow(obs_y - pred_y, 2))/(2 * pow(sigma_y, 2));
                    //Calculate weight using normalization terms and exponent
                    double weight = gauss_norm * exp(-exponent);
                    //Calculate particle's final weight
                    particles[i].weight *= weight;
                }
            }
        }
        //Update weights
        weights[i] = particles[i].weight;
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	//Create a vector<Particle> to save resampled particles.
    vector<Particle> particle_updated;

    default_random_engine gen;

    //Generate random particle indices
    uniform_int_distribution<int> particles_index(0, num_particles-1);
    int index = particles_index(gen);

    //Get maximum weight
    double weight_max = *max_element(weights.begin(), weights.end());

    // uniform random distribution [0.0, max_weight)
    uniform_real_distribution<double> distr_uniform(0.0, 2.0 * weight_max);

    double beta = 0.0;

    //Run the spinning wheel for re-sampling
    for (int i = 0; i < num_particles; i++) {
        beta += distr_uniform(gen);
        while (beta > weights[index]) {
            beta -= weights[index];
            index = (index + 1) % num_particles;
        }
        particle_updated.push_back(particles[index]);
    }

    particles = particle_updated;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
