import {defs, tiny} from './examples/common.js';

const {
    Vector, Vector3, vec, vec3, vec4, color, hex_color, Shader, Matrix, Mat4, Light, Shape, Material, Scene, Texture,
} = tiny;

const {Cube, Axis_Arrows, Textured_Phong} = defs

export class Demo4 extends Scene {
    /**
     *  **Base_scene** is a Scene that can be added to any display canvas.
     *  Setup the shapes, materials, camera, and lighting here.
     */
    constructor() {
        // constructor(): Scenes begin by populating initial values like the Shapes and Materials they'll need.
        super();

        // TODO:  Create two cubes, including one with the default texture coordinates (from 0 to 1), and one with the modified
        //        texture coordinates as required for cube #2.  You can either do this by modifying the cube code or by modifying
        //        a cube instance's texture_coords after it is already created.
        this.shapes = {
            box_1: new Cube(),
            box_2: new Cube(),
            axis: new Axis_Arrows()
        }
        console.log(this.shapes.box_1.arrays.texture_coord)


        // TODO:  Create the materials required to texture both cubes with the correct images and settings.
        //        Make each Material from the correct shader.  Phong_Shader will work initially, but when
        //        you get to requirements 6 and 7 you will need different ones.
        this.materials = {
            phong: new Material(new Textured_Phong(), {
                color: hex_color("#ffffff"),
            }),
            texture: new Material(new Textured_Phong(), {
                color: hex_color("#000000"),  // <-- changed base color to black
                ambient: 1.0,  // <-- changed ambient to 1
                texture: new Texture("assets/debug_texture.jpg")
            }),
            texture_demo: new Material(new Texture_Demo(), {
                color: hex_color("#000000"),  // <-- changed base color to black
                ambient: 1.0,  // <-- changed ambient to 1
                texture: new Texture("assets/debug_texture.jpg")
            }),
        }

        this.initial_camera_location = Mat4.look_at(vec3(0, 10, 20), vec3(0, 0, 0), vec3(0, 1, 0));
    }

    make_control_panel() {
        // TODO:  Implement requirement #5 using a key_triggered_button that responds to the 'c' key.
    }

    display(context, program_state) {
        if (!context.scratchpad.controls) {
            this.children.push(context.scratchpad.controls = new defs.Movement_Controls());
            // Define the global camera and projection matrices, which are stored in program_state.
            program_state.set_camera(Mat4.translation(0, 0, -8));
        }

        program_state.projection_transform = Mat4.perspective(
            Math.PI / 4, context.width / context.height, 1, 100);

        const light_position = vec4(10, 10, 10, 1);
        program_state.lights = [new Light(light_position, color(1, 1, 1, 1), 1000)];

        let t = program_state.animation_time / 1000, dt = program_state.animation_delta_time / 1000;
        let model_transform = Mat4.identity();

        // // Example 1
        // this.box_1_transform = Mat4.translation(-2,0,0);
        // this.box_2_transform = Mat4.translation(2,0,0);
        //
        // // this.shapes.box_2.arrays.texture_coord.forEach(
        // //     (v, i, l) => l[i] = vec(v[1], v[0])
        // // )
        //
        // // equivalent for loop
        // let texture_coord = this.shapes.box_2.arrays.texture_coord;
        // for (let i = 0; i < texture_coord.length; i++) {
        //     let new_coord = vec(texture_coord[i][1], texture_coord[i][0]);
        //     this.shapes.box_2.arrays.texture_coord[i] = new_coord;
        // }
        //
        // this.shapes.box_1.draw(context, program_state, this.box_1_transform, this.materials.texture);
        // this.shapes.box_2.draw(context, program_state, this.box_2_transform, this.materials.texture);

        // // Example 2
        // this.box_1_transform = Mat4.translation(-2,0,0);
        // this.box_2_transform = Mat4.translation(2,0,0);
        // this.shapes.box_2.arrays.texture_coord.forEach(
        //     (v, i, l) => v[0] = v[0] + 0.25
        // )
        // this.shapes.box_1.draw(context, program_state, this.box_1_transform, this.materials.texture);
        // this.shapes.box_2.draw(context, program_state, this.box_2_transform, this.materials.texture);

        // Example 3 & 4
        this.box_1_transform = Mat4.translation(-2,0,0);
        this.box_2_transform = Mat4.translation(2,0,0);
        this.shapes.box_2.arrays.texture_coord.forEach(
            (v, i, l) => v[0] = v[0] + 0.25
        )
        this.shapes.box_1.draw(context, program_state, this.box_1_transform, this.materials.texture_demo);
        this.shapes.box_2.draw(context, program_state, this.box_2_transform, this.materials.texture_demo);
    }
}


class Texture_Demo extends Textured_Phong {
    // TODO:  Modify the shader below (right now it's just the same fragment shader as Textured_Phong) for requirement #6.
    fragment_glsl_code() {
        return this.shared_glsl_code() + `
            varying vec2 f_tex_coord;
            uniform sampler2D texture;
            uniform float animation_time;
            
            void main(){
                // Sample the texture image in the correct place:
                float scale_factor = 1.0 + sin(animation_time) * 0.5;
                vec2 scaled_tex_coord = vec2(f_tex_coord.x, f_tex_coord.y * scale_factor);
                vec4 tex_color = texture2D( texture, scaled_tex_coord);
                
                // black out wrt to the scaled tex corrd
                float u = mod(scaled_tex_coord.x, 1.0);
                float v = mod(scaled_tex_coord.y, 1.0);
                float distance_to_center = sqrt(pow(u - 0.5, 2.0) + pow(v - 0.5, 2.0));
                if (distance_to_center > 0.3 && distance_to_center < 0.4) {
                    tex_color = vec4(0, 0, 0, 1.0);
                }
                
                // // black out wrt to the original tex corrd
                // float u = mod(f_tex_coord.x, 1.0);
                // float v = mod(f_tex_coord.y, 1.0);
                // float distance_to_center = sqrt(pow(u - 0.5, 2.0) + pow(v - 0.5, 2.0));
                // if (distance_to_center > 0.3 && distance_to_center < 0.4) {
                //     tex_color = vec4(0, 0, 0, 1.0);
                // }
                
                if( tex_color.w < .01 ) discard;
                                                                         // Compute an initial (ambient) color:
                gl_FragColor = vec4( ( tex_color.xyz + shape_color.xyz ) * ambient, shape_color.w * tex_color.w ); 
                                                                         // Compute the final color with contributions from lights:
                gl_FragColor.xyz += phong_model_lights( normalize( N ), vertex_worldspace );
        } `;
    }
}