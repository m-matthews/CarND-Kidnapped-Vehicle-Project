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
#include <limits.h>
#include <float.h>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  num_particles = 20;

  // Create normal (Gaussian) distributions for x, y and theta.
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);

  // Update the vector allocation.
  particles.reserve(num_particles);
  weights.reserve(num_particles);

  // Initialise the particles to the specified x, y and theta plus measurement uncertainty.
  for(int p=0; p<num_particles; ++p)
  {
    Particle particle;
    particle.id = p;
    particle.x = dist_x(gen);
    particle.y = dist_y(gen);
    particle.theta = dist_theta(gen);
    particle.weight = 1.0;
    particles.push_back(particle);
    weights.push_back(particle.weight);
  }

  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

  // Create normal (Gaussian) distributions for x, y and theta (zero based).
  normal_distribution<double> dist_x(0.0, std_pos[0]);
  normal_distribution<double> dist_y(0.0, std_pos[1]);
  normal_distribution<double> dist_theta(0.0, std_pos[2]);

  // Update particle positions based on delta_t, velocity and yaw_rate plus measurement uncertainty.
  for(unsigned int p=0; p<num_particles; ++p)
  {
    double theta = particles[p].theta;

    if(fabs(yaw_rate)<0.0001)
    {
      double dv = delta_t*velocity;
      particles[p].x += dv * cos(theta) + dist_x(gen);
      particles[p].y += dv * sin(theta) + dist_y(gen);
      particles[p].theta += dist_theta(gen);
    }
    else
    {
      double dy = delta_t*yaw_rate;
      double vy = velocity/yaw_rate;
      particles[p].x += vy * (sin(theta + dy) - sin(theta)) + dist_x(gen);
      particles[p].y += vy * (cos(theta) - cos(theta + dy)) + dist_y(gen);
      particles[p].theta += dy + dist_theta(gen);
    }
  }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  for(unsigned int o=0; o<observations.size(); ++o)
  {
    double mindist = DBL_MAX;
    observations[o].id = INT_MAX;

    for(unsigned int p=0; p<predicted.size(); ++p)
    {
      double currdist = dist(predicted[p].x, predicted[p].y, observations[o].x, observations[o].y);
      if (currdist<mindist)
      {
        observations[o].id = p; // predicted[p].id; // ^*^ Should be the better via second definition.
        mindist = currdist;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
    std::vector<LandmarkObs> observations, Map map_landmarks) {
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
  double stdx = std_landmark[0];
  double stdy = std_landmark[1];
  std::vector<LandmarkObs> map_observations;
  std::vector<LandmarkObs> predicted;

  for(unsigned int p=0; p<num_particles; ++p)
  {
    double px = particles[p].x;
    double py = particles[p].y;
    double ptheta = particles[p].theta;

    // Convert observations into MAP coordinates.
    map_observations.clear();
    unsigned int obscount = observations.size();
    map_observations.reserve(obscount);
    for(unsigned int o=0; o<obscount; ++o)
    {
      double ox = observations[o].x;
      double oy = observations[o].y;
      LandmarkObs map_observation;
      map_observation.id = observations[o].id;
      map_observation.x = ox * cos(ptheta) - oy * sin(ptheta) + px;
      map_observation.y = ox * sin(ptheta) + oy * cos(ptheta) + py;
      map_observations.push_back(map_observation);
    }

    // Find the Map items that are predicted to be within the current sensor range.
    predicted.clear();
    int map_size = map_landmarks.landmark_list.size();
    for (int i=0; i<map_size; ++i)
    {
      double currdist = dist(px, py, map_landmarks.landmark_list[i].x_f, map_landmarks.landmark_list[i].y_f);
      if (currdist<=sensor_range)
      {
        LandmarkObs observation;
        observation.id = map_landmarks.landmark_list[i].id_i;
        observation.x = map_landmarks.landmark_list[i].x_f;
        observation.y = map_landmarks.landmark_list[i].y_f;
        predicted.push_back(observation);
      }
    }

    dataAssociation(predicted, map_observations);

    // Generate the weights for each particle.
    double weight = 1.0;
    double x, y;
    for(unsigned int o=0; o<obscount; ++o)
    {
      if (map_observations[o].id==INT_MAX)
      {
        x = sensor_range;
        y = sensor_range;
      }
      else
      {
        x = map_observations[o].x - predicted[map_observations[o].id].x;
        y = map_observations[o].y - predicted[map_observations[o].id].y;
      }
      weight *= (1.0/(2.0*M_PI*stdx*stdy)) * exp(-1.0 * ((x*x)/(2.0*stdx*stdx) + (y*y)/(2.0*stdy*stdy)));
    }

    particles[p].weight = weight;
    weights[p] = weight;
  }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

  discrete_distribution<> discdist(weights.begin(), weights.end());

  std::vector<Particle> next_particles;
  next_particles.reserve(num_particles);

  for(unsigned int p=0; p<num_particles; ++p)
  {
    int chosen_particle = discdist(gen);
    next_particles[p] = particles[chosen_particle];
  }

  for(unsigned int p=0; p<num_particles; ++p)
  {
    particles[p] = next_particles[p];
  }
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
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
