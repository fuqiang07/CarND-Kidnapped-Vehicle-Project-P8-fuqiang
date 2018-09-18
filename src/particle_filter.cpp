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

        // init Particles
        Particle.push_back(p);
        weights.push_back(p.weight);
    }
    // Flag, if filter is initialized
    is_initialized = True;
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

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    for (size_t i = 0; i < observations.size(); i++) {
        //Maximum distance can be square root of 2 times the range of sensor.
        double lowest_dist = sensor_range * sqrt(2);
        int closest_landmark_id = -1;
        double obs_x = observations[i].x;
        double obs_y = observations[i].y;

        for (size_t j = 0; j < predicted.size(); j++) {
            double pred_x = predicted[j].x;
            double pred_y = predicted[j].y;
            int pred_id = predicted[j].id;
            double current_dist = dist(obs_x, obs_y, pred_x, pred_y);

            if (current_dist < lowest_dist) {
                lowest_dist = current_dist;
                closest_landmark_id = pred_id;
            }
        }
        observations[i].id = closest_landmark_id;
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
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

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
